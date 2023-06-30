from mini_rec_sys.scorers import BM25Scorer, DenseScorer
from mini_rec_sys.evaluators import Evaluator
from mini_rec_sys.sample_data import get_msmarco_sample_data
from mini_rec_sys.data import Session, SessionDataset, ItemDataset
from mini_rec_sys.encoders import MiniLmEncoder

items, sessions = get_msmarco_sample_data()

# Put document texts into a ItemDataset
item_dataset = ItemDataset(id_name="item_id", data=items, load_fn=lambda x: {"text": x})

# Put relevance judgments into a SessionDataset
collection = SessionDataset(
    id_name="session_id",
    data=sessions,
    item_dataset=item_dataset,
    store_fn=lambda id, row: Session(
        query=row["query"],
        session_id=id,
        positive_items=row["positive_items"],
        positive_relevances=row["relevances"],
        negative_items=row["negative_items"],
    ),
)

# Simple BM25 scorer for reranking
bm25_scorer = BM25Scorer(
    query_key="query",
    test_documents_key="item_attributes",
    train_documents={qid: v for qid, v in iter(item_dataset)},
    fields=["text"],
)

# Evaluate performance
ndcg, se = Evaluator(bm25_scorer, collection).evaluate()
print(f"BM25 reranking ndcg is: {ndcg:.4f}+-{se:.4f}")

# Simple minilm scorer for reranking
minilm_encoder = MiniLmEncoder(dim_embed=384, max_length=64)
minilm_scorer = DenseScorer(
    query_key="query",
    test_documents_key="item_attributes",
    passage_text_key="text",
    q_encoder=minilm_encoder,
    p_encoder=minilm_encoder,
    batch_size=32,
)
ndcg, se = Evaluator(minilm_scorer, collection).evaluate()
print(f"Minilm reranking ndcg is {ndcg:.4f}+-{se:.4f}")
