## Installation

1. Create the Python environment.

```
conda create -n itserr-wp8 -y --no-default-packages python==3.9
conda activate itserr-wp8
```

2. Install Pytorch.

```
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

3. Install faiss.

```
conda install -n itserr-wp8 -y -c conda-forge faiss-cpu
```

4. Clone the repo and install other dependencies.

```
git clone https://github.com/aimagelab/itserr-wp8-latin-embeddings.git
cd itserr-wp8-latin-embeddings
pip install -r requirements.txt
```

5. Install dependencies for the Latin-BERT tokenizer.
```
python -c "from cltk.data.fetch import FetchCorpus; corpus_downloader = FetchCorpus(language='lat'); corpus_downloader.import_corpus('lat_models_cltk')"
```

## Run interactive search
First, login on Hugging Face 🤗 using your read token:
```
huggingface-cli login
```

Now download pre-extracted embeddings from our fine-tuned models for the Latin language.
```
git lfs install
git clone https://huggingface.co/datasets/itserr/WP8-Latin-Embeddings-Indices
```

You can download our fine-tuned models from [Hugging Face 🤗](https://huggingface.co/collections/itserr/wp8-latin-embeddings-67e6bd21c191d487d3cccd64). 

We have released two flavors of [Latin-BERT](https://github.com/dbamman/latin-bert/tree/master) (`--model_type latin_bert`):
- ``Latin-BERT-W_VULG-S_VL``: this model has been fine-tuned with contrastive learning, using correspondencies between the *Vulgate* and *Vetus Latina* translations of the Bible as positive pairs, and in-batch negatives.
- ``Latin-BERT-W_VULG-S_VL-Synt``: this model has been fine-tuned with additional synthetic hard negatives generated with Chat-GPT.

Similarly, we have released two flavors of [LaBERTa](https://huggingface.co/bowphs/LaBerta) (`--model_type laberta`), fine-tuned with the same training recipe of our Latin-BERT:
- ``LaBERTa-W_VULG-S_VL``
- ``LaBERTa-W_VULG-S_VL-Synt``

Run interactive search with the following command. Adjust `--index_root` and `--passages_root` accordingly to the path where you downloaded the pre-extracted embeddings.

```
PYTHONPATH=.:./latin-bert python interactive_search.py \
--index_root ./WP8-Latin-Embeddings-Indices \
--passages_root ./WP8-Latin-Embeddings-Indices \
--model_name_or_path itserr/LaBERTa-W_VULG-S_VL-Synt \
--model_type laberta \
--bible_id W_VULG \
--n_search_items 10
```
You can choose to search among the embeddings extracted from either the Vulgate or the Vetus Latina translations of the Bible, by setting `--bible_id` equal to `W_VULG` or `S_VL` respectively.

