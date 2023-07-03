import pandas as pd
import numpy as np
import math
from torch.utils.data import DataLoader
from mini_rec_sys.scorers import BaseScorer
from mini_rec_sys.data import Session, SessionDataset, BatchedSequentialSampler
from mini_rec_sys.constants import (
    USER_ATTRIBUTES_NAME,
    ITEM_ATTRIBUTES_NAME,
    SESSION_NAME,
)

from pdb import set_trace


def ndcg(relevances: list[float], k=10):
    idcg = idcgk(relevances, k)
    if idcg <= 0.0:
        return 0.0
    return dcgk(relevances, k) / idcg


def dcgk(relevances: list[float], k: int):
    relevances = relevances[:k]  # truncate at k
    return sum(
        [rel / math.log(pos + 2, 2) for pos, rel in enumerate(relevances)]
    )  # +2 as enumerate is 0-indexed


def idcgk(relevances: list[float], k: int):
    optimal = sorted(relevances, reverse=True)
    return dcgk(optimal, k)


def mean_with_se(metrics: list[float]):
    if metrics is None or len(metrics) == 0:
        return None
    return np.mean(metrics), np.std(metrics) / math.sqrt(len(metrics))


class Evaluator:
    """
    Class that takes in a scorer and evaluation data, and return the
    NDCG@K metric for reranking.

    # TODO: Currently we only have scorer as a pipeline, but in future will extend
    to a composable pipeline, e.g. retrieval -> scorer.
    """

    def __init__(
        self,
        pipeline: BaseScorer,
        dataset: SessionDataset = None,
        batch_size: int = 32,
    ) -> None:
        """ """
        self.pipeline = pipeline
        self.dataset = dataset
        self.batch_size = batch_size

    def evaluate(self, k=20, return_raw=False):
        """
        For now we will just evaluate the NDCG, extend in future to take in
        other metrics.
        """
        assert self.dataset is not None, "Dataset must be provided to evaluate"
        metrics = []
        sampler = BatchedSequentialSampler(
            self.dataset, batch_size=self.batch_size, drop_last=False
        )
        for batch in DataLoader(
            self.dataset, batch_sampler=sampler, collate_fn=lambda x: x
        ):
            metrics.extend(self.evaluate_batch(batch, k=k, return_raw=True))
        assert len(metrics) == len(self.dataset)
        if return_raw:
            return metrics
        return mean_with_se(metrics)

    def evaluate_batch(self, batch: list[dict], k=20, return_raw=False):
        metrics = []
        scores: list[list[float]] = self.pipeline(batch)
        for i, row in enumerate(batch):
            row_scores = scores[i]
            session: Session = row["session"]
            relevances_reranked = self.rerank(row_scores, session.relevances)
            metrics.append(ndcg(relevances_reranked, k=k))
        assert len(metrics) == len(batch)
        if return_raw:
            return metrics
        return mean_with_se(metrics)

    def rerank(self, scores: list[float], items: list[object]):
        """
        Rerank items in descending score order.
        """
        sorting = np.argsort(-np.array(scores))
        return np.array(items)[sorting].tolist()
