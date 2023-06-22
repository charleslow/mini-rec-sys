from mini_rec_sys.scorers import BM25Scorer


class TestBM25Scorer:
    default_documents = {
        1: {"title": "mouse", "text": "i am a mouse, i like cheese."},
        2: {"title": "cat", "text": "i am cat. i like to eat mouse."},
        3: {"title": "cheese", "text": "i am cheese. cheezy cheese."},
    }

    def test_build_and_score(self):
        test_docs = sorted(self.default_documents.items(), key=lambda x: x[0])
        input_data = {"query": "mouse", "docs": [doc[1] for doc in test_docs]}
        scorer = BM25Scorer(
            "query", "docs", self.default_documents, fields=["title", "text"]
        )
        scores = scorer.score_single(input_data)
        assert scores[0] > scores[1] > scores[2], "Score order not correct."
