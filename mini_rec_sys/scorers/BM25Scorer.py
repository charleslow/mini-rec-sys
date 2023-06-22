import math
import re
import numpy as np
from collections import Counter
from mini_rec_sys.scorers import BaseScorer
from pdb import set_trace

STOP_WORDS = [
    "i",
    "am",
    "stop",
    "the",
    "to",
    "and",
    "a",
    "in",
    "it",
    "is",
    "I",
    "that",
    "had",
    "on",
    "for",
    "were",
    "was",
]
MIN_SCORE = -99999.0


class BM25Scorer(BaseScorer):
    """
    A scorer model that trains a BM25F model on a set of item texts.
    At test time, it scores documents on the fly by vectorizing them and
    scoring similarity between query and item based on the BM25 text
    similarity score.
    """

    def __init__(
        self,
        query_key: str,
        test_documents_key: str,
        train_documents: dict[str, dict],
        fields: list[str],
        params: dict[str, object] = None,
        field_weights: dict[str, int] = None,
    ):
        """
        query_key: at test time, key containing the query
        test_documents_key: at test time, key containing the test documents
        train_documents: training documents to train the model
        fields: fields in each document that will be used
        params: BM25 params
        field_weights: weight to place on each field for the BM25 score
        """
        super().__init__(cols=[query_key, test_documents_key])
        self.query_key = query_key
        self.test_documents_key = test_documents_key
        self.fields = fields if isinstance(fields, list) else [fields]
        self.set_field_weights(field_weights)
        self.set_params(params)
        self.non_alphabets = re.compile(r"[^a-zA-Z\s]")
        self.stop_words = set(STOP_WORDS)

        # Initialize empty dicts for training
        self.df = {}  # Document frequency of a term
        self.field_doclens = {field: 0.0 for field in fields}
        self.train(train_documents)

    def score(self, input_data: dict | list[dict]):
        if input_data is None:
            return None
        if isinstance(input_data, dict):
            return self.score_single(input_data)
        return [self.score_single(row) for row in input_data]

    def score_single(self, row: dict):
        """Generate score for one row of input_data."""
        query = row[self.query_key]
        test_docs = row[self.test_documents_key]
        qterms = list(set(self.doc_to_words(query)))
        matched_terms = [qterm for qterm in qterms if qterm in self.df.keys()]
        k1 = self.params["k1"]

        scores = []
        for doc in test_docs:
            if doc is None:
                scores.append(MIN_SCORE)
                continue

            score = 0.0
            doc_terms = {field: self.tokenize(doc[field]) for field in self.fields}
            doc_lens = {
                field: self.tf_to_doclen(doc_terms[field]) for field in self.fields
            }

            for qterm in matched_terms:
                tf_overall = 0.0
                for field in self.fields:
                    bf = self.params["b"][field]
                    wf = self.field_weights[field]
                    doclen = doc_lens[field]

                    if qterm in doc_terms[field]:
                        tf = doc_terms[field][qterm]
                        doclen_term = (1 - bf) + bf * doclen / self.field_doclens[field]
                        tf_overall += wf * tf / doclen_term
                df_t = self.df.get(qterm, 0.0)
                idf = math.log(1 + (self.N - df_t + 0.5) / (df_t + 0.5))
                score += idf * tf_overall / (k1 + tf_overall)
            scores.append(score)
        return scores

    def tf_to_doclen(self, tf_dict: dict[str, int]):
        """
        Transform a dict of {term: term_frequency} into bm25 doclen.
        """
        if len(tf_dict) == 0:
            return 0.0
        tfs = [math.pow(math.log(tf), 2) for tf in tf_dict.values()]
        return math.sqrt(sum(tfs))

    def tokenize(self, text):
        """
        Use fitted values to transform a text into a dict of term to tf.
        """
        text = self.doc_to_words(text)
        terms = {}
        for word, tf in Counter(text).items():
            terms[word] = tf
        return terms

    def doc_to_words(self, text):
        """
        Tokenizes a text into term tokens. Removes all html tags <xxx>, and keeps only alphabets.
        """
        if (text is None) or (text == ""):
            return []

        text = re.sub("<[^<]+?>", "", text)  # Remove all html tags like <xx>
        text = text.replace("/", " ")  # Replace / with space
        text = text.replace("\n", " ")  # Replace \n with space
        text = text.replace("-", " ")  # Replace - with space
        text = re.sub(self.non_alphabets, "", text)
        text = text.lower()
        return [w for w in text.split() if not w in self.stop_words]

    def train(self, documents):
        """
        Compute and store term/document data, then build the inverted_list.
        """
        self.N = len(documents)

        for _, d in documents.items():
            word_set = set()

            for field in self.fields:
                doclen_score = 0.0
                text = d[field]
                words = self.doc_to_words(text)
                for word, tf in Counter(words).items():
                    doclen_score += math.pow(math.log(tf) + 1, 2)
                    word_set.add(word)

                doclen_score = math.sqrt(doclen_score)
                self.field_doclens[field] += doclen_score

            if len(word_set) == 0:
                continue

            # Add to document frequency
            for w in word_set:
                self.df[w] = self.df.get(w, 0) + 1

        # Average the field doclens
        for field in self.fields:
            self.field_doclens[field] /= self.N
        print(f"Processed {self.N} documents.")

    def set_params(self, d: dict = None):
        """Params for BM25F search"""
        if d is None:
            d = {
                "k1": 1.2,
                "b": {field: 0.7 for field in self.fields},
            }
        else:
            assert "k1" in d
            assert "b" in d
            for field in self.fields:
                assert field in d["b"]
        self.params = d

    def set_field_weights(self, d: dict = None):
        if d is None:
            d = {field: 1.0 for field in self.fields}
        else:
            for field in self.fields:
                assert field in d
        self.field_weights = d
