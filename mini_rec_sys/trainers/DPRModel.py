"""
Module for training a query and passage encoder based on Karpukhin 2020 -
Dense Passage Retrieval for Open-Domain Question Answering.
"""
from __future__ import annotations
from torch.optim import Optimizer, Adam
import random

from mini_rec_sys.data import Session, ItemDataset, SessionDataset, Sampler
from mini_rec_sys.trainers.base_trainers import BaseModel
from mini_rec_sys.encoders import BaseBertEncoder
from mini_rec_sys.scorers import DenseScorer
from mini_rec_sys.evaluators import Evaluator
from mini_rec_sys.constants import (
    ITEM_ATTRIBUTES_NAME,
    VAL_METRIC_NAME,
    TEST_METRIC_NAME,
)
from mini_rec_sys.utils import shuffle, WeightedSampler

import torch
import torch.nn.functional as F

from pdb import set_trace


class DPRModel(BaseModel):
    """
    Takes in a list of (query, positive_text) pairs, and produces
    a similarity matrix:
                            S = Q dot P^T
    Let B denote batch size and d the embedding dimension.
    where Q (B x d) and P (B x d) are query and job matrices respectively,
    and S will be a (B x B) similarity matrix.

    S[i, j] represents a positive pair if i = j, and negative otherwise.
    """

    def __init__(
        self,
        query_key: str,
        item_text_key: str,
        q_encoder: BaseBertEncoder,
        p_encoder: BaseBertEncoder,
        train_dataset: SessionDataset,
        sampler: Sampler,
        model_params: dict = None,
        optimizer_class: Optimizer = Adam,
        learning_rate: float = 1e-5,
        check_for_conflict: bool = True,
        val_dataset: SessionDataset = None,
        val_batch_size: int = 32,
        val_method: str = "default",
        val_k: int = 20,
        val_metric_name: str = VAL_METRIC_NAME,
        test_dataset: SessionDataset = None,
        test_batch_size: int = 32,
        test_method: str = "default",
        test_k: int = 20,
        test_metric_name: str = TEST_METRIC_NAME,
    ) -> None:
        super().__init__(
            train_dataset=train_dataset,
            sampler=sampler,
            model_params=model_params,
            optimizer_class=optimizer_class,
            learning_rate=learning_rate,
            val_dataset=val_dataset,
            val_batch_size=val_batch_size,
            test_dataset=test_dataset,
            test_batch_size=test_batch_size,
        )
        self.query_key = query_key
        self.item_text_key = item_text_key
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder
        self.check_for_conflict = check_for_conflict
        assert val_method in ["default", "rerank"]
        self.val_method = val_method
        self.val_k = val_k
        self.val_metric_name = val_metric_name
        assert test_method in ["default", "rerank"]
        self.test_method = test_method
        self.test_k = test_k

    def forward(self, batch: list[dict]):
        triplets = self.load_triplets(batch)
        S = self.compute_S(triplets)
        return self.compute_loss(S)

    def load_triplets(self, batch: list[dict]):
        """
        From the mini batch, sample a triplet of (query, positive_text, negative_text)
        1. For each query, sample a positive passage
        2. (Optional) For each query, sample a negative passage

        Return as:
            a list[tuple] of (query, positive_passage, negative_passage)
            or a list[tuple] of (query, positive_passage).
        """
        triplets = []
        queries = set()
        already_sampled_items = set()

        for d in batch:
            session: Session = d["session"]

            if session.query in queries:
                continue
            queries.add(session.query)

            # Sample positive item, ensuring that it has not appeared as a
            # positive or negative item before in the batch
            FLAG_POSITIVE = False
            positive_weights = (
                [1] * len(session.positive_items)
                if not session.positive_weights
                else session.positive_weights
            )
            positive_sampler = WeightedSampler(
                {
                    item: weight
                    for item, weight in zip(session.positive_items, positive_weights)
                },
                pop=True,
            )
            while len(positive_sampler) > 0:
                positive_item = positive_sampler.sample()
                if self.check_for_conflict and (positive_item in already_sampled_items):
                    continue
                positive_text = self.train_dataset.load_item(positive_item).get(
                    self.item_text_key, None
                )
                if positive_text is not None:
                    FLAG_POSITIVE = True
                    break

            if not FLAG_POSITIVE:
                continue

            # Keep track of positive items for queries in the batch
            for item in session.positive_items:
                already_sampled_items.add(item)

            if not self.train_dataset.has_negative_items:
                triplets.append(
                    (
                        getattr(session, self.query_key),
                        positive_text,
                    )
                )
                continue

            # Sample negative item, ensuring that it has not appeared as a
            # positive or negative item before in the batch
            FLAG_NEGATIVE = False
            negative_weights = (
                [1] * len(session.negative_items)
                if not session.negative_items
                else session.negative_weights
            )
            negative_sampler = WeightedSampler(
                {
                    item: weight
                    for item, weight in zip(session.negative_items, negative_weights)
                },
                pop=True,
            )
            while len(negative_sampler) > 0:
                negative_item = negative_sampler.sample()
                if self.check_for_conflict and (negative_item in already_sampled_items):
                    continue
                negative_text = self.train_dataset.load_item(negative_item).get(
                    self.item_text_key, None
                )
                if negative_text is not None:
                    FLAG_NEGATIVE = True
                    already_sampled_items.add(negative_item)
                    break

            if not FLAG_NEGATIVE:
                continue

            triplets.append(
                (getattr(session, self.query_key), positive_text, negative_text)
            )

        return triplets

    def compute_S(self, triplets: list[tuple]):
        """
        Each item in the list is a pair or triplet:
        - Query
        - Positive text
        - Negative text (optional)
        """
        # Encode queries
        queries = [trip[0] for trip in triplets]
        Q = self.q_encoder(queries)

        # Encode positive samples
        positive_texts = [trip[1] for trip in triplets]
        P = self.p_encoder(positive_texts)

        # Encode negative job samples
        if self.train_dataset.has_negative_items:
            negative_texts = [trip[2] for trip in triplets]
            P2 = self.p_encoder(negative_texts)
            P = torch.vstack([P, P2])  # 2*n_batch x embed_dim

        S = Q @ P.T
        return S

    def compute_loss(self, S: torch.Tensor):
        """
        Compute negative log likelihood loss from the B x B or B x 2B
        similarity matrix produced by the forward method.
        Return the mean of the row-wise softmax loss of the positive pair on the
        diagonal of the similarity matrix.
        """
        return -F.log_softmax(S, dim=1).trace() / len(S)

    def validation_step(self, batch: list[dict], batch_idx: int):
        if self.val_method == "default":
            return super().validation_step(batch, batch_idx)
        elif self.val_method == "rerank":
            return self.rerank_validation(batch, batch_idx)

    def rerank_validation(self, batch: list[dict], batch_idx: int):
        # Use a DenseScorer to rerank and submit ndcg
        scorer = DenseScorer(
            query_key=self.query_key,
            test_documents_key=ITEM_ATTRIBUTES_NAME,
            passage_text_key=self.item_text_key,
            q_encoder=self.q_encoder,
            p_encoder=self.p_encoder,
            batch_size=self.val_batch_size,
        )
        ndcg, se = Evaluator(scorer).evaluate_batch(
            batch, k=self.val_k, return_raw=False
        )
        self.log(self.val_metric_name, ndcg, prog_bar=True, batch_size=len(batch))
        return ndcg

    def test_step(self, batch: list[dict], batch_idx: int):
        if self.test_method == "default":
            return super().test_step(batch, batch_idx)
        elif self.test_method == "rerank":
            return self.rerank_test(batch, batch_idx)

    def rerank_test(self, batch: list[dict], batch_idx: int):
        # Use a DenseScorer to rerank and submit ndcg
        scorer = DenseScorer(
            query_key=self.query_key,
            test_documents_key=ITEM_ATTRIBUTES_NAME,
            passage_text_key=self.item_text_key,
            q_encoder=self.q_encoder,
            p_encoder=self.p_encoder,
            batch_size=self.test_batch_size,
        )
        metrics = Evaluator(scorer).evaluate_batch(
            batch, k=self.test_k, return_raw=True
        )
        self.test_metrics.extend(metrics)
