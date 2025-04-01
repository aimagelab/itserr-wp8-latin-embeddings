from dataclasses import dataclass
from typing import List, Optional, Union, Dict
from transformers.modeling_outputs import ModelOutput
import torch
from torch import Tensor
import torch.distributed as dist
import pandas as pd
from logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ContrastiveTrainingModelOutput(ModelOutput):
    r_at_1: Optional[Tensor] = None


def labels_from_passage_ids(
    passage_ids: List[str],
    positive_mask: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    assert len(passage_ids) == len(positive_mask)
    # assuming there is 1! positive passage per query
    Nq, Np = positive_mask.sum().item(), positive_mask.size(0)
    labels = torch.zeros(Nq, Np, dtype=dtype)
    pids = pd.Series(passage_ids)
    pid2idxs = dict(list(pids.groupby(pids, sort=False)))
    i = 0
    for is_pos, pid in zip(positive_mask, passage_ids):
        if is_pos:
            labels[i, pid2idxs[pid].index] = 1
            i += 1
    assert i == Nq
    labels /= labels.sum(dim=1, keepdim=True)
    return labels


def compute_recall(
    eval_preds,
    query_features: Union[Tensor, List[Tensor]],
    passage_features: Union[Tensor, List[Tensor]],
    labels: Union[Tensor, List[Tensor]] = None,
    passage_ids: Optional[Union[List[str], List[List[str]]]] = None,
    positive_mask: Optional[Union[Tensor, List[Tensor]]] = None,
    matrix_batch_size: Optional[int] = None,
    K: List[int] = [1, 2, 3, 5, 10, 20, 50, 100]
):
    if isinstance(query_features, list):
        query_features = torch.cat(query_features, dim=0)
    if isinstance(passage_features, list):
        passage_features = torch.cat(passage_features, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)
    if isinstance(positive_mask, list):
        positive_mask = torch.cat(positive_mask, dim=0)
    if isinstance(passage_ids[0], list):
        passage_ids = [y for x in passage_ids for y in x]
    
    rank = dist.get_rank()
    WS = dist.get_world_size()
    dtype = passage_features.dtype
    all_query_bsz = [None] * WS
    dist.all_gather_object(all_query_bsz, query_features.size(0))
    all_passage_bsz = [None] * WS
    dist.all_gather_object(all_passage_bsz, passage_features.size(0))
    all_positive_mask = [torch.empty(all_passage_bsz[i], dtype=torch.long) for i in range(WS)]
    dist.all_gather(all_positive_mask, positive_mask)
    all_passage_ids = [None] * WS
    dist.all_gather_object(all_passage_ids, passage_ids)
    all_passage_features = [torch.empty((all_passage_bsz[i], passage_features.size(1)), dtype=dtype) for i in range(WS)]
    dist.all_gather(all_passage_features, passage_features.to(rank))

    all_positive_mask = torch.cat(all_positive_mask, dim=0)
    all_passage_features = torch.cat(all_passage_features, dim=0).to(rank)
    all_passage_ids = [y for x in all_passage_ids for y in x]
    assert len(all_positive_mask) == len(all_passage_features)
    query_features = query_features.to(rank)

    if labels is None:
        labels = labels_from_passage_ids(
            passage_ids=all_passage_ids, 
            positive_mask=all_positive_mask, 
            dtype=dtype
        )

    if matrix_batch_size:
        rbsz = cbsz = matrix_batch_size
    else:
        rbsz = query_features.size(0)
        cbsz = all_passage_features.size(0)
    Nq, Np = query_features.size(0), all_passage_features.size(0)
    
    sim_scores = torch.zeros(Nq, Np, dtype=dtype, device=rank)
    logger.info(f"[RANK {rank}] Computing {Nq}x{Np} similarity matrix with block size {rbsz}x{cbsz} ...")
    for r in range(0, Nq, rbsz):
        Q = query_features[r:r + rbsz]
        for c in range(0, Np, cbsz):
            P = all_passage_features[c:c + cbsz]
            sim_scores[r:r + rbsz, c:c + cbsz] = Q @ P.T
    del query_features
    del all_passage_features

    ranking = sim_scores.argsort(dim=1, descending=True, stable=True)
    recall = torch.empty(len(K), dtype=torch.float32, device=rank)
    labels = labels.to(rank)
    for i, k in enumerate(K):
        recall[i] = (labels.gather(dim=1, index=ranking[:, :k]) > 0).any(dim=1).sum()
    dist.reduce(recall, dst=0, op=dist.ReduceOp.SUM)
    if rank:
        ret = {}
    else:
        N = sum(all_query_bsz)
        recall = recall.cpu() / N
        ret = {f"R@{k}": recall[i].item() for i, k in enumerate(K)}
    return ret
    