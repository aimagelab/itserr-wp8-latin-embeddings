import torch
import argparse
import torch.utils
import logging_utils
import pandas as pd
import numpy as np
from pathlib import Path
import json
import faiss
import torch.nn.functional as F
from models import RetrieverModel, tokenize
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter

logger = logging_utils.get_logger(__name__)


def load_index(index_path: Path, experiment_name: str):
    if not isinstance(index_path, Path):
        index_path = Path(index_path)
    index_path = index_path.joinpath(experiment_name)
    embds_path = index_path.joinpath('embeddings.npy')
    embds = np.load(embds_path)
    logger.info(
        f"Loaded embeddings with shape {embds.shape} from `{embds_path}`")
    index = faiss.IndexFlatL2(embds.shape[1])
    index.add(embds)
    logger.info('Built `faiss.IndexFlatL2` index')

    keys_path = index_path.joinpath('keys.json')
    with open(keys_path, 'r') as f:
        keys = json.load(f)

    return index, keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--model_type', type=str,
                        choices=['latin_bert', 'laberta'])
    parser.add_argument('--bible_id', type=str,
                        choices=['W_VULG', 'S_VL'])
    parser.add_argument('--passages_root', type=str),
    parser.add_argument('--index_root', type=str),
    parser.add_argument('--n_search_items', type=int, default=10)
    args = parser.parse_args()

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    logger.info(
        f"Using device `{device if device == 'cpu' else torch.cuda.get_device_name(device)}`")

    model = RetrieverModel.from_pretrained(
        args.model_name_or_path, device_map=device)
    model.init_tokenizer()
    model.eval()
    tokenizer = model.tokenizer

    jsonl_passages_path = Path(args.passages_root).joinpath(f"{args.bible_id}__passages.jsonl")
    passages = pd.read_json(jsonl_passages_path, lines=True)
    logger.info(
        f"Read {len(passages)} passages from `{jsonl_passages_path}`")
    
    model_type = args.model_type
    if '-synt' in args.model_name_or_path.lower():
        model_type += '_synt'
    index, index_keys = load_index(args.index_root, f"{model_type}__{args.bible_id}__mean")

    try:
        while True:
            query = input('(Press CTRL-D to exit)\nQuery > ')
            inputs = tokenize([query], tokenizer)
            inputs = (x.to(device) for x in inputs)
            with torch.inference_mode():
                query_embeds = model.get_embeddings(*inputs)
            query_embeds = F.normalize(query_embeds, p=2, dim=-1).cpu().numpy()
            index.search(query_embeds, args.n_search_items)
            distance, rank = index.search(query_embeds, args.n_search_items)
            distance, rank = (distance[0], rank[0])
            pids = [index_keys[idx] for idx in rank]
            results = [dict(id=pid, rank=i, distance=str(distance[i]), passage=passages.loc[passages.passage_id ==
                            pid].iloc[0].content) for i, pid in enumerate(pids)]
            results = dict(query=query, results=results)
            results = json.dumps(results, indent=2)
            results = highlight(results, JsonLexer(), TerminalFormatter())
            print(results)
            print(f"\n\n{'*' * 50}\n\n")
    except EOFError:
        logger.info('Exiting...')
