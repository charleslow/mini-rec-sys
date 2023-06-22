from mini_rec_sys.data import Loader


class TestLoader:
    default_documents = {
        1: {"title": "mouse", "text": "i am a mouse, i like cheese."},
        2: {"title": "cat", "text": "i am cat. i like to eat mouse."},
        3: {"title": "cheese", "text": "i am cheese. cheezy cheese."},
    }

    def test_loading(self):
        loader = Loader(data=self.default_documents)
        for i in self.default_documents:
            assert loader.load(i) == self.default_documents[i]

    def test_load_function(self):
        loader = Loader(data=self.default_documents, load_fn=lambda x: x["title"])
        for i in self.default_documents:
            assert loader.load(i) == self.default_documents[i]["title"]
