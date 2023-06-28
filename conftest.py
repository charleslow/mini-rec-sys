import pytest


@pytest.fixture
def default_documents():
    return {
        1: {"title": "mouse", "text": "i am a mouse, i like cheese."},
        2: {"title": "cat", "text": "i am cat. i like to eat mouse."},
        3: {"title": "cheese", "text": "i am cheese. cheezy cheese."},
    }


@pytest.fixture
def default_session_data():
    return {
        "session1": dict(
            positive_items=[1, 2],
            positive_relevances=[2, 1],
            negative_items=[3],
            query="mouse",
        ),
        "session2": dict(
            positive_items=[2],
            positive_relevances=[2],
            negative_items=[1],
            query="cat",
        ),
    }
