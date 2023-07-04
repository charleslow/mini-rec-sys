from mini_rec_sys.data import (
    ItemDataset,
    SessionDataset,
    Session,
    BatchedSequentialSampler,
)
from mini_rec_sys.encoders import MiniLmEncoder
from mini_rec_sys.trainers import DPRModel, train


def test_dpr_model(default_documents, default_session_data, capsys):
    item_dataset = ItemDataset(
        id_name="item_id",
        data=default_documents,
    )
    session_dataset = SessionDataset(
        id_name="session_id",
        data=default_session_data,
        store_fn=lambda id, row: Session(
            session_id=id,
            positive_items=row["positive_items"],
            positive_relevances=row["positive_relevances"],
            negative_items=row["negative_items"],
            query=row["query"],
        ),
        item_dataset=item_dataset,
    )
    train_dataset = session_dataset.split_dataset(lambda x: hash(x) % 2 == 0)
    val_dataset = session_dataset.split_dataset(lambda x: hash(x) % 2 != 0)
    encoder = MiniLmEncoder(dim_embed=384, max_length=20)
    model = DPRModel(
        query_key="query",
        item_text_key="text",
        q_encoder=encoder,
        p_encoder=encoder,
        train_dataset=train_dataset,
        sampler=BatchedSequentialSampler(train_dataset, 5, drop_last=False),
        val_dataset=val_dataset,
        val_batch_size=5,
    )
    train(
        model,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
    )
