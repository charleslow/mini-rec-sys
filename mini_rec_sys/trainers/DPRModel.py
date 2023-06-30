"""
Module for training a query and passage encoder based on Karpukhin 2020 -
Dense Passage Retrieval for Open-Domain Question Answering.
"""
from torch.optim import Optimizer, Adam
import random

from mini_rec_sys.data import Session, ItemDataset, SessionDataset, Sampler
from mini_rec_sys.trainers.base_trainers import BaseModel
from mini_rec_sys.encoders import BaseBertEncoder
from mini_rec_sys.scorers import DenseScorer
from mini_rec_sys.evaluators import Evaluator

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
        val_dataset: SessionDataset = None,
        val_batch_size: int = 32,
        validation_method: str = "default",
    ) -> None:
        super().__init__(
            train_dataset=train_dataset,
            sampler=sampler,
            model_params=model_params,
            optimizer_class=optimizer_class,
            learning_rate=learning_rate,
            val_dataset=val_dataset,
            val_batch_size=val_batch_size,
        )
        self.query_key = query_key
        self.item_text_key = item_text_key
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder
        assert validation_method in ["default", "rerank"]
        self.validation_method = validation_method

    def forward(self, batch: list[dict]):
        triplets = self.load_triplets(batch)
        S = self.compute_S(triplets)
        return self.compute_loss(S)

    def load_triplets(self, batch: list[dict]):
        triplets = []
        for d in batch:
            session: Session = d["session"]
            positive_item = random.sample(session.positive_items, k=1)[0]
            if self.train_dataset.has_negative_items:
                negative_item = random.sample(session.negative_items, k=1)[0]
            triplets.append(
                (
                    getattr(session, self.query_key),
                    self.train_dataset.load_item(positive_item)[self.item_text_key],
                    self.train_dataset.load_item(negative_item)[self.item_text_key],
                )
            )
        return triplets

    def compute_S(self, triplets: list[tuple]):
        """
        Each item in the list is a pair or triplet:
        - Query
        - Positive text
        - Negative text (optional)
        """
        # Check if negative samples are provided
        negative_sampling = len(triplets[0]) == 3

        # Encode queries
        queries = [trip[0] for trip in triplets]
        Q = self.q_encoder(queries)

        # Encode positive samples
        positive_texts = [trip[1] for trip in triplets]
        P = self.p_encoder(positive_texts)

        # Encode negative job samples
        if negative_sampling:
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

    def validation_step(self, batch: list[dict], batch_idx):
        if self.validation_method == "default":
            return super().validation_step(batch, batch_idx)
        elif self.validation_method == "rerank":
            return self.rerank_validation(batch, batch_idx)

    def rerank_validation(self, batch: list[dict], batch_idx):
        # Use a DenseScorer to rerank and submit ndcg
        scorer = DenseScorer(
            query_key=self.query_key,
            test_documents_key="item_attributes",
            passage_text_key=self.item_text_key,
            q_encoder=self.q_encoder,
            p_encoder=self.p_encoder,
            batch_size=self.val_batch_size,
        )
        Evaluator()
