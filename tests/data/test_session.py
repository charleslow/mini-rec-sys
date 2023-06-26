from mini_rec_sys.data import Session
import pytest
from pydantic.dataclasses import ValidationError


class TestSession:
    def test_valid_session_does_not_raise_error(self):
        session = Session(
            session_id="123",
            positive_items=["a", "b"],
            positive_relevances=[1, 2],
        )

    def test_differing_relevances_and_positive_items_length_raises_error(self):
        with pytest.raises(ValidationError):
            session = Session(
                session_id="123",
                positive_items=["a", "b"],
                positive_relevances=[1, 2, 3],
            )

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            session = Session(session_id="123", positive_items=["a"])
        with pytest.raises(ValidationError):
            session = Session(session_id="123", positive_relevances=[1, 2])
        with pytest.raises(TypeError):
            session = Session(positive_items=["a"], positive_relevances=[1])
