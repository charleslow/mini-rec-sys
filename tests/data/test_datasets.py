from mini_rec_sys.data import Session, SessionDataset
from mini_rec_sys.data.datasets import Dataset
from pydantic.dataclasses import ValidationError
import pytest

from pdb import set_trace


class TestDataset:
    def test_loading(self, default_documents):
        dataset = Dataset(data=default_documents)
        for i in default_documents:
            assert dataset.load(i) == default_documents[i]

    def test_load_function(self, default_documents):
        dataset = Dataset(data=default_documents, load_fn=lambda x: x["title"])
        for i in default_documents:
            assert dataset.load(i) == default_documents[i]["title"]

    def test_iterator(self, default_documents):
        dataset = Dataset(data=default_documents)
        items = set([v["text"] for k, v in iter(dataset)])
        assert items == set([v["text"] for v in default_documents.values()])


class TestSessionDataset:
    def test_session_dataset_init_no_errors(self, default_session_data):
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
        )

    def test_session_dataset_must_load_sessions(self):
        with pytest.raises(AssertionError):
            dataset = SessionDataset(data={"a": 1}, store_fn=lambda id, row: row)
