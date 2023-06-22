from mini_rec_sys.scorers import BM25Scorer
from mini_rec_sys.evaluators import Evaluator
from mini_rec_sys.data import SessionCollection, Loader, Session
import json

SESSIONS_FILE = "../sample_data/msmarco_reranking/msmarco_sample_sessions.json"
ITEMS_FILE = "../sample_data/msmarco_reranking/msmarco_sample_items.json"

with open(SESSIONS_FILE) as f:
    sessions = json.load(f)
item_loader = Loader(id_name="item_id", data=ITEMS_FILE, load_fn=lambda x: {"text": x})
collection = SessionCollection(
    sessions=[
        Session(
            session_id=qid,
            positive_items=v["positive_items"],
            positive_relevances=v["relevances"],
            negative_items=v["negative_items"],
        )
        for qid, v in sessions.items()
    ],
    item_loader=item_loader,
)
bm25_scorer = BM25Scorer(
    query_key="query",
    test_documents_key="item_attributes",
    train_documents={qid: {"text": v} for qid, v in iter(item_loader)},
    fields=["text"],
)
ndcg, se = Evaluator(bm25_scorer, collection).evaluate()
print(f"BM25 reranking ndcg is: {ndcg:.4f}+-{se:.4f}")
