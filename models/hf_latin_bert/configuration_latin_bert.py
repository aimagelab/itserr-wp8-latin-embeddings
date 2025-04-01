from typing import Optional
from transformers import BertConfig


class HfLatinBertConfig(BertConfig):
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        tokenizer_path: Optional[str] = None,
        **kwargs
    ):
        self._is_latin_bert = True
        super().__init__(**kwargs)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path