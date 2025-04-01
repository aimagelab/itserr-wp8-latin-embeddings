from .configuration_retriever import RetrieverConfig, BertPoolingStrategy
from .modeling_retriever import RetrieverModel, RetrieverModelOutput, pool_bert_output
from transformers import AutoConfig, AutoModel

AutoConfig.register("retriever", RetrieverConfig)
AutoModel.register(RetrieverConfig, RetrieverModel)