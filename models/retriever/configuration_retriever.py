from typing import Optional
from transformers import PretrainedConfig, AutoConfig, BertConfig
from enum import Enum
from logging_utils import get_logger

CLIP_LOGIT_SCALE_INIT_VALUE = 2.6592
CLIP_LOGIT_SCALE_PRETRAINED_VALUE = 4.605170249938965

logger = get_logger(__name__)


class BertPoolingStrategy(str, Enum):
    CLS = "cls"
    CLS_TANH = "cls_tanh"
    MEAN = "mean"
    MEAN_NO_CLS = "mean_w_cls"
    L2NORM_SUM = "l2norm_sum"


class RetrieverConfig(PretrainedConfig):
    model_type = "retriever"

    def __init__(
        self,
        embedding_config=None,
        pooling_strategy: Optional[BertPoolingStrategy] = BertPoolingStrategy.CLS,
        logit_scale_init_value: Optional[float] = CLIP_LOGIT_SCALE_INIT_VALUE,
        contrastive_topk: int = 0,
        sample_contrastive_topk: str = None,
        **kwargs
    ):
        if embedding_config is None:
            logger.warning("`embedding_config` is None, default to BertConfig")
            embedding_config = BertConfig(**kwargs)
        elif embedding_config.get('_is_latin_bert', False):
            embedding_config = BertConfig(**embedding_config)
        else:
            embedding_config = AutoConfig.from_pretrained(embedding_config.get('name_or_path', embedding_config.get('_name_or_path')), **embedding_config)

        super().__init__(**kwargs)

        self.embedding_config = embedding_config
        self.pooling_strategy = pooling_strategy
        self.logit_scale_init_value = logit_scale_init_value
        self.contrastive_topk = contrastive_topk
        self.sample_contrastive_topk = sample_contrastive_topk
