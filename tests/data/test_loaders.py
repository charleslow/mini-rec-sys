from mini_rec_sys.data import Loader, SessionLoader, Session
from pydantic.dataclasses import ValidationError
import pytest

from pdb import set_trace


class TestLoader:
    def test_loading(self, default_documents):
        loader = Loader(data=default_documents)
        for i in default_documents:
            assert loader.load(i) == default_documents[i]

    def test_load_function(self):
        loader = Loader(data=self.default_documents, load_fn=lambda x: x["title"])
        for i in self.default_documents:
            assert loader.load(i) == self.default_documents[i]["title"]

    def test_iterator(self):
        loader = Loader(data=self.default_documents)
        items = set([v["text"] for k, v in iter(loader)])
        assert items == set([v["text"] for v in self.default_documents.values()])


class TestSessionLoader:
    sessions = {
        1: Session(
            session_id=1,
            positive_items=[1, 2],
            positive_relevances=[2, 1],
            negative_items=[3],
            query="mouse",
        )
    }

    def test_session_loader_init_no_errors(self):
        loader = SessionLoader(data=self.sessions)

    def test_session_loader_must_load_sessions(self):
        with pytest.raises(AssertionError):
            loader = SessionLoader(data={"a": 1})
