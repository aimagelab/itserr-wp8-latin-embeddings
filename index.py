from functools import partial
import torch
import torch.distributed as dist
import argparse
from tqdm import tqdm
import torch.utils
import logging_utils
import pandas as pd
import numpy as np
from pathlib import Path
import json
from models import RetrieverModel, tokenize, BertPoolingStrategy
import torch.nn.functional as F

logger = logging_utils.get_logger(__name__)


class PassagesDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, args: argparse.Namespace):
        self.passages = pd.read_json(args.passages_jsonl_path, lines=True)
        if args.n_samples:
            self.passages = self.passages.head(args.n_samples)
        logger.info(f"Loaded {len(self.passages)} passages from `{data_path}`")

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        return self.passages.iloc[idx].to_dict()


def collate_fn(examples, tokenizer=None, args: argparse.Namespace = None):
    ret = {k: [ex[k] for ex in examples] for k in examples[0].keys()}
    ret['input_ids'], ret['attention_mask'] = tokenize(ret['content'], tokenizer)
    return ret


def get_model_and_tokenizer(args, device):
    if args.model_name_or_path is not None:
        model = RetrieverModel.from_pretrained(args.model_name_or_path)
    else:
        model = RetrieverModel.from_pretrained_embedding_model(args.embedding_model_name_or_path)
    model.init_tokenizer()
    if args.use_average_embeddings:
        model.set_pooling_strategy(BertPoolingStrategy.MEAN)
    elif args.use_l2_norm_sum_l2_norm:
        model.set_pooling_strategy(BertPoolingStrategy.L2NORM_SUM)
    model.eval()
    model.to(device)
    return model, model.tokenizer


def get_embeddings(model, batch):
    input_ids = batch['input_ids'].to(model.device)
    attention_mask = batch['attention_mask'].to(model.device)
    return model.get_embeddings(input_ids, attention_mask)


def get_model_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--embedding_model_name_or_path', type=str)
    parser.add_argument('--use_average_embeddings', action='store_true')
    parser.add_argument('--use_l2_norm_sum_l2_norm', action='store_true')
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_model_args_parser()])
    parser.add_argument('--passages_jsonl_path', type=str)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--index_path', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataloader_num_workers', type=int, default=0)
    args = parser.parse_args()

    device = 'cpu'
    if torch.cuda.is_available():
        dist.init_process_group(backend='nccl')
        device = dist.get_rank()
    logger.info(
        f"Using device `{device if device == 'cpu' else torch.cuda.get_device_name(device)}`")

    index_path = Path(args.index_path)
    index_path.mkdir(parents=True, exist_ok=True, mode=0o775)
    index_path = index_path.joinpath(args.experiment_name)
    index_path.mkdir(parents=True, exist_ok=True, mode=0o775)
    index_output_path = index_path.joinpath('embeddings.npy')
    keys_output_path = index_path.joinpath('keys.json')

    model, tokenizer = get_model_and_tokenizer(args, device)

    dataset = PassagesDataset(args.passages_jsonl_path, args)
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        sampler=torch.utils.data.distributed.DistributedSampler(dataset),
        collate_fn=partial(collate_fn, tokenizer=tokenizer, args=args)
    )
    dataloader = torch.utils.data.DataLoader(**dataloader_kwargs)

    passage_ids = []
    passage_embds = []
    for batch in tqdm(dataloader, desc='Extracting embeddings ...'):
        with torch.inference_mode():
            embds = get_embeddings(model, batch)
        embds = F.normalize(embds, p=2, dim=-1).cpu().numpy()
        passage_ids.extend(batch['passage_id'])
        passage_embds.append(embds)

    passage_embds = np.concatenate(passage_embds, axis=0)
    np.save(index_output_path, passage_embds)
    logger.info(
        f"Saved {len(passage_embds)} embeddings to `{index_output_path}`")

    with open(keys_output_path, 'w') as f:
        json.dump(passage_ids, f)
    logger.info(f"Saved keys to `{keys_output_path}`")
