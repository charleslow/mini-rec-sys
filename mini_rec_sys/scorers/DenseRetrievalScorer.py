from typing import List, Tuple
import torch
import numpy as np
from tqdm import tqdm

from dual_encoder.utils import Batcher
from dual_encoder.encoders import BaseBertEncoder


class DenseRetrievalScorer:
    """
    A scorer that uses a query encoder to embed the query and passage encoder
    to encode the passages / items. Scores are then generated based on cosine
    similarity between the query embedding and passage embeddings.
    """

    def __init__(
        self, q_encoder: BaseBertEncoder, p_encoder: BaseBertEncoder, batch_size: int
    ) -> None:
        """
        q_encoder: query encoder
        p_encoder: passage / item encoder
        batch_size: how many queries or items to embed in a batch, based on
            memory considerations
        """
        self.q_encoder = q_encoder
        self.q_encoder.eval()
        self.p_encoder = p_encoder
        self.p_encoder.eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def q_encode(self, texts: List[str]):
        return self.q_encoder(texts).cpu().detach().numpy()

    @torch.no_grad()
    def p_encode(self, texts: List[str]):
        return self.p_encoder(texts).cpu().detach().numpy()

    @torch.no_grad()
    def score(self, test_data: List[Tuple]):
        """Generate list of scores for each row of test_data.

        test_data: List of tuples of the following form:
            (query_string, list_of_item_texts)

        The score list contains scores corresponding to the similarity match
        between the query_string and each item in the list_of_item_texts.
        """
        self.p_encoder.eval()

        # Encode queries
        queries = [d[0] for d in test_data]
        q_embed = []
        for batch in tqdm(Batcher(queries, self.batch_size).batches()):
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
        for tup in test_data:
            inner_list = []
            for item_text in tup[1]:
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
        for batch in tqdm(Batcher(item_texts, self.batch_size).batches()):
            p_embed.append(self.p_encode(batch))
        p_embed = np.vstack(p_embed)  # n_unique_jobs x embed_dim

        # Score and extract scores to rank
        S = q_embed @ p_embed.T  # n_batch x n_unique_jobs
        results = []
        for i, idxs in enumerate(idx_list):
            row_scores = S[i, idxs]
            results.append(row_scores.tolist())
        return results
