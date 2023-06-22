import pandas as pd
import numpy as np
import math
from mini_rec_sys.scorers import BaseScorer
from mini_rec_sys.data import SessionCollection, Session

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
        collection: SessionCollection,
    ) -> None:
        """ """
        self.pipeline = pipeline
        self.collection = collection

    def evaluate(self, k=20):
        """
        For now we will just evaluate the NDCG, extend in future to take in
        other metrics.

        TODO: Add batching.
        """
        metrics = []
        for session in self.collection.sessions:
            scores = self.score_session(session)
            relevances_reranked = self.rerank(scores, session.relevances)
            metrics.append(ndcg(relevances_reranked, k=k))
        return mean_with_se(metrics)

    def score_session(self, session: Session, k=20):
        item_attributes = (
            session.items
            if self.collection.item_loader is None
            else self.collection.load_items(session.items)
        )
        user_attributes = (
            session.user
            if self.collection.user_loader is None
            else self.collection.load_users(session.user)
        )
        scores = self.pipeline(
            {
                "query": session.query,
                "user_attributes": user_attributes,
                "item_attributes": item_attributes,
            }
        )
        return scores

    def rerank(self, scores: list[float], items: list[object]):
        """
        Rerank items in descending score order.
        """
        sorting = np.argsort(-np.array(scores))
        return np.array(items)[sorting].tolist()
