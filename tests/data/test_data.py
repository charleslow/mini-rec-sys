from mini_rec_sys.data import Session, User, Item
import pytest
from pydantic.dataclasses import ValidationError


class TestSession:
    def test_valid_session_does_not_raise_error(self):
        session = Session(
            session_id="123",
            positive_items=[Item("a"), Item("b")],
            relevances=[1, 2],
        )

    def test_differing_relevances_and_positive_items_length_raises_error(self):
        session = Session(
            session_id="123",
            positive_items=[Item("a"), Item("b")],
            relevances=[1, 2],
        )

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            session = Session(session_id="123", positive_items=[Item("a")])
        with pytest.raises(ValidationError):
            session = Session(session_id="123", relevances=[1, 2])
        with pytest.raises(TypeError):
            session = Session(positive_items=[Item("a")], relevances=[1])
