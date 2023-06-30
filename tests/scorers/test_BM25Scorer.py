from mini_rec_sys.scorers import BM25Scorer
from pdb import set_trace


class TestBM25Scorer:
    def test_build_and_score(self, default_documents):
        test_docs = sorted(default_documents.items(), key=lambda x: x[0])
        input_data = {"query": "mouse", "docs": [doc[1] for doc in test_docs]}
        scorer = BM25Scorer(
            "query", "docs", default_documents, fields=["title", "text"]
        )
        scores = scorer.score_single(input_data)
        assert scores[0] > scores[1] > scores[2], "Score order not correct."
