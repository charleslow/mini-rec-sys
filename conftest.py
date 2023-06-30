import pytest
import random

DEFAULT_DOCUMENTS = {
    1: {"title": "mouse", "text": "i am a mouse, i like cheese."},
    2: {"title": "cat", "text": "i am cat. i like to eat mouse."},
    3: {"title": "cheese", "text": "i am cheese. cheezy cheese."},
    4: {"title": "dog", "text": "i am dog. woof woof."},
    5: {"title": "house", "text": "i am house. roof roof."},
}


@pytest.fixture
def default_documents() -> dict:
    return DEFAULT_DOCUMENTS


@pytest.fixture
def default_session_data() -> dict:
    n = 50
    data = {}
    for i in range(n):
        pos_item, neg_item = random.sample(list(DEFAULT_DOCUMENTS), k=2)
        words = DEFAULT_DOCUMENTS[pos_item]["text"].split()
        data[f"session_{i+1}"] = dict(
            positive_items=[pos_item],
            negative_items=[neg_item],
            positive_relevances=[1],
            query=random.sample(words, k=1)[0],
        )
    return data


@pytest.fixture
def minilm_encoder():
    from mini_rec_sys.encoders import MiniLmEncoder

    return MiniLmEncoder(dim_embed=384, max_length=20)
