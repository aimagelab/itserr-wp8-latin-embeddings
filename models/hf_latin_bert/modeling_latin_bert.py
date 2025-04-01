import torch
from transformers import BertModel
from .configuration_latin_bert import HfLatinBertConfig
from tensor2tensor.data_generators import text_encoder
from cltk.tokenizers.lat.lat import LatinWordTokenizer as WordTokenizer
from cltk.tokenizers.lat.lat import LatinPunktSentenceTokenizer as SentenceTokenizer
import numpy as np
import torch


class LatinTokenizer():
	def __init__(self, encoder):
		self.vocab={}
		self.reverseVocab={}
		self.encoder=encoder

		self.vocab["[PAD]"]=0
		self.vocab["[UNK]"]=1
		self.vocab["[CLS]"]=2
		self.vocab["[SEP]"]=3
		self.vocab["[MASK]"]=4

		for key in self.encoder._subtoken_string_to_id:
			self.vocab[key]=self.encoder._subtoken_string_to_id[key]+5
			self.reverseVocab[self.encoder._subtoken_string_to_id[key]+5]=key

	def convert_tokens_to_ids(self, tokens):
		wp_tokens=[]
		for token in tokens:
			if token == "[PAD]":
				wp_tokens.append(0)
			elif token == "[UNK]":
				wp_tokens.append(1)
			elif token == "[CLS]":
				wp_tokens.append(2)
			elif token == "[SEP]":
				wp_tokens.append(3)
			elif token == "[MASK]":
				wp_tokens.append(4)

			else:
				wp_tokens.append(self.vocab[token])

		return wp_tokens

	def tokenize(self, text):
		tokens=text.split(" ")
		wp_tokens=[]
		for token in tokens:

			if token in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}:
				wp_tokens.append(token)
			else:

				wp_toks=self.encoder.encode(token)

				for wp in wp_toks:
					wp_tokens.append(self.reverseVocab[wp+5])

		return wp_tokens


def get_sentence_mask(sents):
	mask = []
	sentence_tokenizer  = SentenceTokenizer()
	for i, sent in enumerate(sents):
		subsents = sentence_tokenizer.tokenize(sent)
		mask.extend(i for _ in subsents)
	return np.array(mask, dtype=np.uint32)


def convert_to_toks(sents, truncate=True, max_len=512):

	word_tokenizer = WordTokenizer()

	all_sents=[]

	for data in sents:
		text=data.lower()
		
		# sents=sent_tokenizer.tokenize(text)
		for sent in [text]:
			tokens=word_tokenizer.tokenize(sent)
			filt_toks=[]
			filt_toks.append("[CLS]")
			for tok in tokens:
				if tok != "":
					filt_toks.append(tok)
			if truncate:
				filt_toks = filt_toks[:max_len - 1]
			filt_toks.append("[SEP]")

			all_sents.append(filt_toks)

	return all_sents


def get_batches(sentences, max_batch, tokenizer):

		maxLen=0
		for sentence in sentences:
			length=0
			for word in sentence:
				toks=tokenizer.tokenize(word)
				length+=len(toks)

			if length> maxLen:
				maxLen=length

		all_data=[]
		all_masks=[]
		all_labels=[]
		all_transforms=[]

		for sentence in sentences:
			tok_ids=[]
			input_mask=[]
			labels=[]
			transform=[]

			all_toks=[]
			n=0
			for idx, word in enumerate(sentence):
				toks=tokenizer.tokenize(word)
				all_toks.append(toks)
				n+=len(toks)

			cur=0
			for idx, word in enumerate(sentence):
				toks=all_toks[idx]
				ind=list(np.zeros(n))
				for j in range(cur,cur+len(toks)):
					ind[j]=1./len(toks)
				cur+=len(toks)
				transform.append(ind)

				tok_ids.extend(tokenizer.convert_tokens_to_ids(toks))

				input_mask.extend(np.ones(len(toks)))
				labels.append(1)

			all_data.append(tok_ids)
			all_masks.append(input_mask)
			all_labels.append(labels)
			all_transforms.append(transform)

		lengths = np.array([len(l) for l in all_data])

		# Note sequence must be ordered from shortest to longest so current_batch will work
		ordering = np.argsort(lengths)
		
		ordered_data = [None for i in range(len(all_data))]
		ordered_masks = [None for i in range(len(all_data))]
		ordered_labels = [None for i in range(len(all_data))]
		ordered_transforms = [None for i in range(len(all_data))]
		

		for i, ind in enumerate(ordering):
			ordered_data[i] = all_data[ind]
			ordered_masks[i] = all_masks[ind]
			ordered_labels[i] = all_labels[ind]
			ordered_transforms[i] = all_transforms[ind]

		batched_data=[]
		batched_mask=[]
		batched_labels=[]
		batched_transforms=[]

		i=0
		current_batch=max_batch

		while i < len(ordered_data):

			batch_data=ordered_data[i:i+current_batch]
			batch_mask=ordered_masks[i:i+current_batch]
			batch_labels=ordered_labels[i:i+current_batch]
			batch_transforms=ordered_transforms[i:i+current_batch]

			max_len = max([len(sent) for sent in batch_data])
			max_label = max([len(label) for label in batch_labels])

			for j in range(len(batch_data)):
				
				blen=len(batch_data[j])
				blab=len(batch_labels[j])

				for k in range(blen, max_len):
					batch_data[j].append(0)
					batch_mask[j].append(0)
					for z in range(len(batch_transforms[j])):
						batch_transforms[j][z].append(0)

				for k in range(blab, max_label):
					batch_labels[j].append(-100)

				for k in range(len(batch_transforms[j]), max_label):
					batch_transforms[j].append(np.zeros(max_len))

			batched_data.append(torch.LongTensor(batch_data))
			batched_mask.append(torch.FloatTensor(batch_mask))
			batched_labels.append(torch.LongTensor(batch_labels))
			# batched_transforms.append(torch.FloatTensor(batch_transforms))

			bsize=torch.FloatTensor(batch_transforms).shape
			
			i+=current_batch

		return batched_data, batched_mask, batched_transforms, ordering


def get_batches_unsorted(sentences, max_batch, tokenizer):

		maxLen=0
		for sentence in sentences:
			length=0
			for word in sentence:
				toks=tokenizer.tokenize(word)
				length+=len(toks)

			if length> maxLen:
				maxLen=length

		all_data=[]
		all_masks=[]
		all_transforms=[]

		for sentence in sentences:
			tok_ids=[]
			input_mask=[]
			labels=[]
			transform=[]

			all_toks=[]
			n=0
			for idx, word in enumerate(sentence):
				toks=tokenizer.tokenize(word)
				all_toks.append(toks)
				n+=len(toks)

			cur=0
			for idx, word in enumerate(sentence):
				toks=all_toks[idx]
				ind=list(np.zeros(n))
				for j in range(cur,cur+len(toks)):
					ind[j]=1./len(toks)
				cur+=len(toks)
				transform.append(ind)

				tok_ids.extend(tokenizer.convert_tokens_to_ids(toks))

				input_mask.extend(np.ones(len(toks)))
				labels.append(1)

			pad_len = maxLen - len(tok_ids)
			tok_ids.extend(np.zeros(pad_len, dtype=np.int32))
			input_mask.extend(np.zeros(pad_len, dtype=np.int32))

			all_data.append(np.array(tok_ids))
			all_masks.append(np.array(input_mask))
			all_transforms.append(transform)

		batched_data = torch.from_numpy(np.stack(all_data).astype(np.int64))
		batched_mask = torch.from_numpy(np.stack(all_masks).astype(np.int64))
		batched_transforms = []

		return batched_data, batched_mask, batched_transforms
	

def load_latin_bert_tokenizer(tokenizer_path: str):
    return LatinTokenizer(text_encoder.SubwordTextEncoder(tokenizer_path))


class HfLatinBertModel(BertModel):
    config_class = HfLatinBertConfig
    base_model_prefix = "hf_latin_bert"

    def __init__(self, config: HfLatinBertConfig):
        super().__init__(config)
        self.tokenizer = load_latin_bert_tokenizer(config.tokenizer_path)

    def get_berts(self, raw_sents, batch_size=32):
        sents = convert_to_toks(raw_sents)
        batched_data, batched_mask, batched_transforms = get_batches_unsorted(
            sents, batch_size, self.tokenizer)
        outputs = super(HfLatinBertModel, self).bert_forward(
            input_ids=batched_data, attention_mask=batched_mask)
        return outputs, batched_mask
