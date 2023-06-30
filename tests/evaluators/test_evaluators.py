from mini_rec_sys.evaluators import Evaluator
from mini_rec_sys.scorers import BM25Scorer, DenseScorer
from mini_rec_sys.data import Session, SessionDataset, ItemDataset
from mini_rec_sys.constants import ITEM_ATTRIBUTES_NAME

from pdb import set_trace


class TestEvaluator:
    session_data = {
        "session_1": dict(
            positive_items=[1, 2],
            positive_relevances=[2, 1],
            negative_items=[3],
            query="mouse",
        )
    }

    def test_evaluation_is_correct(self, default_documents, minilm_encoder):
        bm25_scorer = BM25Scorer(
            query_key="query",
            test_documents_key="item_attributes",
            train_documents=default_documents,
            fields=["title", "text"],
        )
        dense_scorer = DenseScorer(
            query_key="query",
            test_documents_key=ITEM_ATTRIBUTES_NAME,
            passage_text_key="title",
            q_encoder=minilm_encoder,
            p_encoder=minilm_encoder,
            batch_size=10,
        )
        item_dataset = ItemDataset(id_name="item_id", data=default_documents)
        dataset = SessionDataset(
            id_name="session_id",
            store_fn=lambda id, row: Session(
                session_id=id,
                positive_items=row["positive_items"],
                negative_items=row["negative_items"],
                positive_relevances=row["positive_relevances"],
                query=row["query"],
            ),
            data=self.session_data,
            item_dataset=item_dataset,
        )

        for scorer in [bm25_scorer, dense_scorer]:
            evaluator = Evaluator(pipeline=scorer, dataset=dataset)
            scores = evaluator.score_sessions([dataset.load_session_dict("session_1")])
            assert len(scores) == 1
            ndcg, se = evaluator.evaluate()
            assert ndcg == 1.0

    def test_evaluating_multiple_sessions(
        self, default_documents, default_session_data
    ):
        bm25_scorer = BM25Scorer(
            "query",
            "item_attributes",
            train_documents=default_documents,
            fields=["title", "text"],
        )
        item_dataset = ItemDataset(id_name="item_id", data=default_documents)
        dataset = SessionDataset(
            id_name="session_id",
            store_fn=lambda id, row: Session(
                session_id=id,
                positive_items=row["positive_items"],
                negative_items=row["negative_items"],
                positive_relevances=row["positive_relevances"],
                query=row["query"],
            ),
            data=default_session_data,
            item_dataset=item_dataset,
        )
        evaluator = Evaluator(pipeline=bm25_scorer, dataset=dataset)
        metrics = evaluator.evaluate(k=20, return_raw=True)
        assert len(metrics) == len(default_session_data)
        ndcg, se = evaluator.evaluate()
