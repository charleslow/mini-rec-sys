from mini_rec_sys.scorers import DenseScorer
from mini_rec_sys.encoders import BaseBertEncoder
from pdb import set_trace


class TestDenseScorer:
    def test_build_and_score(
        self, default_documents: dict, minilm_encoder: BaseBertEncoder
    ):
        test_docs = sorted(default_documents.items(), key=lambda x: x[0])
        input_data = {"query": "mouse", "docs": [doc[1] for doc in test_docs]}
        scorer = DenseScorer(
            query_key="query",
            test_documents_key="docs",
            passage_text_key="title",
            q_encoder=minilm_encoder,
            p_encoder=minilm_encoder,
            batch_size=10,
        )
        scores = scorer.score(input_data)
        assert scores[0] > scores[1] > scores[2], "Score order not correct."
