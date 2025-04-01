from dataclasses import dataclass
from typing import Optional, Union
from transformers import PreTrainedModel, AutoModel, AutoTokenizer, AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from models.hf_latin_bert.modeling_latin_bert import HfLatinBertModel
from models.modeling_contrastive_training import ContrastiveTrainingModelOutput
from .configuration_retriever import RetrieverConfig, BertPoolingStrategy
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

@dataclass
class RetrieverModelOutput(ContrastiveTrainingModelOutput):
    loss: Optional[torch.Tensor] = None
    query_features: Optional[torch.Tensor] = None
    query_last_hidden_state: Optional[torch.Tensor] = None
    passage_features: Optional[torch.Tensor] = None
    passage_last_hidden_state: Optional[torch.Tensor] = None


def unpad_tensor(tensors, mask, op=nn.Identity(), skip_first=False):
    # assuming tensors is a batched tensor of shape (N, ...)
    N = len(tensors)
    mask = mask.bool()
    start = 1 if skip_first else 0
    return torch.stack([op(tensors[i, mask[i]][start:]) for i in range(N)])


def pool_bert_output(
    pooling_strategy,
    outputs: BaseModelOutputWithPooling,
    attention_mask: Optional[torch.Tensor] = None
):
    if pooling_strategy == BertPoolingStrategy.CLS_TANH:
        pooler_output = outputs.pooler_output
    elif pooling_strategy == BertPoolingStrategy.CLS:
        pooler_output = outputs.last_hidden_state[:, 0]
    elif pooling_strategy == BertPoolingStrategy.MEAN:
        pooler_output = outputs.last_hidden_state[:,
                                                  1:] * attention_mask[:, 1:, None]
        pooler_output = pooler_output.sum(
            dim=1) / attention_mask[:, 1:].sum(dim=1, keepdim=True)
    elif pooling_strategy == BertPoolingStrategy.L2NORM_SUM:
        def op(x): return F.normalize(x, p=2, dim=-1).sum(dim=0)
        pooler_output = unpad_tensor(
            outputs.last_hidden_state, attention_mask, op=op, skip_first=True)
    else:
        raise ValueError(f"Invalid pooling strategy: {pooling_strategy}")
    outputs['pooler_output'] = pooler_output


class RetrieverModel(PreTrainedModel):
    config_class = RetrieverConfig

    def __init__(
        self,
        config: RetrieverConfig,
        embedding_model: Optional[PreTrainedModel] = None
    ):
        super().__init__(config)
        if embedding_model is None:
            if getattr(config.embedding_config, '_is_latin_bert', False):
                embedding_model = HfLatinBertModel(config.embedding_config)
            else:
                embedding_model = AutoModel.from_config(config.embedding_config)
        self.embedding_model = embedding_model

        if config.logit_scale_init_value is None:
            self.register_buffer(
                "logit_scale", torch.tensor([0.0]))
        else:
            self.logit_scale = nn.Parameter(
                torch.tensor(config.logit_scale_init_value))

        self.tokenizer = getattr(self.embedding_model, "tokenizer", None)

    @classmethod
    def from_pretrained_embedding_model(
        cls,
        embedding_model_name_or_path: str,
        **config_kwargs
    ):
        embd_config = AutoConfig.from_pretrained(embedding_model_name_or_path)
        if getattr(embd_config, '_is_latin_bert', False):
            auto_cls = HfLatinBertModel
        else:
            auto_cls = AutoModel
        del embd_config
        embd_model = auto_cls.from_pretrained(embedding_model_name_or_path)
        config = RetrieverConfig(embd_model.config.to_dict(), **config_kwargs)
        return cls(config, embd_model)

    def init_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.embedding_config.name_or_path)

    def set_pooling_strategy(self, ps: Union[BertPoolingStrategy, str]):
        setattr(self.config, "pooling_strategy", BertPoolingStrategy(ps))

    @property
    def pooling_strategy(self):
        return self.config.pooling_strategy

    def get_logit_scale(self):
        return self.logit_scale.exp()

    def embedding_model_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> BaseModelOutputWithPooling:
        return self.embedding_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        outputs = self.embedding_model_forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pool_bert_output(self.pooling_strategy, outputs, attention_mask)
        return outputs.pooler_output

    def forward(
        self,
        query_input_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        passage_input_ids: Optional[torch.Tensor] = None,
        passage_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        embedding_model_kwargs: Optional[dict] = {},
        **kwargs
    ) -> RetrieverModelOutput:
        query_features = self.get_embeddings(
            input_ids=query_input_ids, attention_mask=query_attention_mask, **embedding_model_kwargs)
        passage_features = self.get_embeddings(
            input_ids=passage_input_ids, attention_mask=passage_attention_mask, **embedding_model_kwargs)

        query_features = F.normalize(query_features, p=2, dim=-1)
        passage_features = F.normalize(passage_features, p=2, dim=-1)

        logit_scale = self.get_logit_scale()
        q_logits = (query_features @ passage_features.T) * logit_scale
        loss_fn = nn.CrossEntropyLoss()

        if self.config.contrastive_topk:
            # TODO: we assume that there are no duplicated positives in the global batch, so no need to compute the argmax
            # pidxs = labels.argmax(dim=1, keepdim=True)
            pidxs = torch.arange(query_features.size(0), device=query_features.device, dtype=torch.long)[:, None]

            # add arguments
            if self.config.sample_contrastive_topk == 'sample':
                sampled_elements = random.sample(range(q_logits.shape[-1]), self.config.contrastive_topk)
                nidxs = q_logits.argsort(dim=1, descending=True)[:, sampled_elements]
            else:
                nidxs = q_logits.argsort(dim=1, descending=True)[:, :self.config.contrastive_topk]
                
            idxs = torch.cat([pidxs, nidxs], dim=1)
            q_logits = q_logits.gather(dim=1, index=idxs)
            labels = torch.zeros((query_features.size(0), q_logits.size(1)), device=query_features.device, dtype=query_features.dtype)
            duplicate_mask = pidxs == idxs
            labels[duplicate_mask] = 1
            labels /= duplicate_mask.sum(dim=1, keepdim=True)

        loss = loss_fn(q_logits, labels)

        preds = q_logits.argmax(dim=-1)
        r_at_1 = torch.stack([labels[i, p] > 0 for i, p in enumerate(preds)]).to(
            torch.float32).mean().item()

        outputs = RetrieverModelOutput(
            loss=loss,
            query_features=query_features,
            passage_features=passage_features,
            r_at_1=r_at_1
        )

        return outputs
