from typing import List
from .retriever import RetrieverModel, BertPoolingStrategy
from .hf_latin_bert import LatinTokenizer, convert_to_toks, get_batches_unsorted
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = None
    embedding_model_name_or_path: Optional[str] = None
    pooling_strategy: str = BertPoolingStrategy.CLS
    logit_scale_init_value: Optional[float] = None
    contrastive_topk: int = 0


def tokenize(text: List[str], tokenizer):
    if isinstance(tokenizer, LatinTokenizer):
        bsz = len(text)
        input_ids = convert_to_toks(text)
        input_ids, attention_mask, _ = get_batches_unsorted(
            input_ids, bsz, tokenizer)
    else:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
    return input_ids, attention_mask


def build_model_module(args: ModelArguments):
    assert not (args.model_name_or_path and args.embedding_model_name_or_path), "Only one of `model_name_or_path` and `embedding_model_name_or_path` should be provided"
    if args.model_name_or_path:
        model = RetrieverModel.from_pretrained(args.model_name_or_path)
    else:
        model = RetrieverModel.from_pretrained_embedding_model(
            args.embedding_model_name_or_path,
            logit_scale_init_value=args.logit_scale_init_value,
            contrastive_topk=args.contrastive_topk,
            sample_contrastive_topk=args.sample_contrastive_topk,
        )

    model.init_tokenizer()
    model.set_pooling_strategy(args.pooling_strategy)

    return dict(
        model=model,
        tokenizer=model.tokenizer
    )