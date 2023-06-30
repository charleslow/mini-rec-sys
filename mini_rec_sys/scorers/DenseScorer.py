from typing import List, Tuple
import torch
import numpy as np
from tqdm import tqdm

from mini_rec_sys.encoders import BaseBertEncoder
from mini_rec_sys.scorers import BaseScorer
from mini_rec_sys.utils import batcher

from pdb import set_trace


class DenseScorer(BaseScorer):
    """
    A scorer that uses a query encoder to embed the query and passage encoder
    to encode the passages / items. Scores are then generated based on cosine
    similarity between the query embedding and passage embeddings.
    """

    def __init__(
        self,
        query_key: str,
        test_documents_key: str,
        passage_text_key: str,
        q_encoder: BaseBertEncoder,
        p_encoder: BaseBertEncoder,
        batch_size: int = 32,
    ) -> None:
        super().__init__(cols=[query_key, test_documents_key, passage_text_key])
        self.q_encoder = q_encoder
        self.p_encoder = p_encoder
        self.batch_size = batch_size
        self.query_key = query_key
        self.test_documents_key = test_documents_key
        self.passage_text_key = passage_text_key

    @torch.no_grad()
    def q_encode(self, texts: list[str]):
        return self.q_encoder(texts).cpu().detach().numpy()

    @torch.no_grad()
    def p_encode(self, texts: list[str]):
        return self.p_encoder(texts).cpu().detach().numpy()

    @torch.no_grad()
    def score(self, test_data: dict | list[dict]):
        """Generate list of scores for each row of test_data.

        The score list contains scores corresponding to the similarity match
        between the query_string and each item in the list_of_item_texts.
        """
        self.q_encoder.eval()
        self.p_encoder.eval()

        # Handle single row of data
        if isinstance(test_data, list):
            queries = [d[self.query_key] for d in test_data]
            item_lists = [d[self.test_documents_key] for d in test_data]
        else:
            queries = [test_data[self.query_key]]
            item_lists = [test_data[self.test_documents_key]]

        # Encode queries
        q_embed = []
        for batch in batcher(queries, self.batch_size):
            q_embed.append(self.q_encode(batch))
        q_embed = np.vstack(q_embed)  # n_batch x embed_dim

        # As there may be duplicate items across the batch, we set up a dict
        # of hash(item_text): idx and only encode each item once to
        # save compute.
        #
        # The idx_list keeps track of the item order in each row of data for
        # looking up the scores later.
        idx_list = []
        hash2idx = {}
        item_texts = []
        idx = 0
        for items in item_lists:
            inner_list = []
            for item in items:
                item_text = item[self.passage_text_key]
                if item_text is None:
                    item_text = ""
                text_hash = hash(item_text)
                if not text_hash in hash2idx:
                    item_texts.append(item_text)
                    hash2idx[text_hash] = idx
                    idx += 1
                inner_list.append(hash2idx[text_hash])
            idx_list.append(inner_list)

        # Batch encode all job texts
        p_embed = []
        for batch in batcher(item_texts, self.batch_size):
            p_embed.append(self.p_encode(batch))
        p_embed = np.vstack(p_embed)  # n_unique_jobs x embed_dim

        # Score and extract scores to rank
        S = q_embed @ p_embed.T  # n_batch x n_unique_jobs
        results = []
        for i, idxs in enumerate(idx_list):
            row_scores = S[i, idxs]
            results.append(row_scores.tolist())

        if isinstance(test_data, list):
            return results
        else:
            return results[0]
