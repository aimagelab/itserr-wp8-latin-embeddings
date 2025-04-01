#!/bin/bash

source activate itserr-wp8
cd localpath/itserr_wp8

export PYTHONPATH=.:./latin-bert
export CUDA_VISIBLE_DEVICES=0

python -u interactive_search.py \
--index_root ./WP8-Latin-Embeddings-Indices \
--passages_root ./WP8-Latin-Embeddings-Indices \
--model_name_or_path itserr/LaBERTa-W_VULG-S_VL-Synt \
--model_type laberta \
--bible_id S_VL \
--n_search_items 5
