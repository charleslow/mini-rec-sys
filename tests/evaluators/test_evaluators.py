from mini_rec_sys.evaluators import Evaluator
from mini_rec_sys.scorers import BM25Scorer
from mini_rec_sys.data import Session, SessionCollection, Loader


class TestEvaluator:
    default_documents = {
        1: {"title": "mouse", "text": "i am mouse mouse, i like cheese."},
        2: {"title": "cat", "text": "i am cat. i like to eat mouse."},
        3: {"title": "avocado", "text": "i am avocado. babobabo."},
    }
    collection = SessionCollection(
        sessions=[
            Session(
                session_id=1,
                positive_items=[1, 2],
                positive_relevances=[2, 1],
                negative_items=[3],
                query="mouse",
            )
        ],
        item_loader=Loader(data=default_documents),
    )
    bm25_scorer = BM25Scorer(
        "query",
        "item_attributes",
        train_documents=default_documents,
        fields=["title", "text"],
    )

    def test_evaluation_is_correct(self):
        evaluator = Evaluator(self.bm25_scorer, self.collection)
        scores = evaluator.score_session(self.collection.sessions[0])
        ndcg, se = evaluator.evaluate()
        assert ndcg == 1.0
