import pandas as pd
import json
from tqdm import tqdm
from importlib import resources

from pdb import set_trace

# inputs
queries_file = "orcas-doctrain-queries.tsv"  # ~10 million rows
candidates_file = "orcas-doctrain-top100"  # ~1 billion rows, sorted
qrels_file = "orcas-doctrain-qrels.tsv"  # ~20 million rows
docs_file = "msmarco-docs.tsv"

# outputs
sessions_file = "msmarco_sample_sessions.json"
items_file = "msmarco_sample_items.json"


def generate_and_write_data():
    SAMPLE_ROWS = 500_000  # how many rows to sample from each file
    N = 50  # number of sessions
    N_NEGATIVE = 10  # number of negative samples per query
    N_WORDS = 50  # cap at number of words per document

    # queries
    queries = pd.read_csv(queries_file, sep="\t", header=None, nrows=SAMPLE_ROWS)
    queries.columns = ["qid", "query"]
    queries = {row["qid"]: row["query"] for _, row in queries.iterrows()}

    # qrels: get relevant docids per query
    qrels = pd.read_csv(qrels_file, sep=" ", header=None, nrows=SAMPLE_ROWS)
    qrels.columns = ["qid", "q", "docid", "relevance"]
    qrels_dict = {}
    for _, row in qrels.iterrows():
        qid = row["qid"]
        qrels_dict[qid] = qrels_dict.get(qid, []) + [row["docid"]]

    # candidates: get impressed docids per query
    candidates = pd.read_csv(candidates_file, sep=" ", header=None, nrows=SAMPLE_ROWS)
    candidates.columns = ["qid", "q", "docid", "rank", "score", "retriever"]
    candidates_dict = {}
    for _, row in candidates.iterrows():
        qid = row["qid"]
        candidates_dict[qid] = candidates_dict.get(qid, []) + [row["docid"]]

    # clean up into sessions
    sessions = {}
    for qid, positive_items in qrels_dict.items():
        if not qid in queries:
            continue
        if not qid in candidates_dict:
            continue
        query = queries[qid]
        impressions = candidates[candidates["qid"] == qid]["docid"].to_list()
        sessions[qid] = {
            "query": query,
            "positive_items": positive_items,
            "relevances": [1] * len(positive_items),
            "negative_items": list(set(impressions) - set(positive_items))[:N_NEGATIVE],
        }
        if len(sessions) >= N:
            break
    with open(sessions_file, "w") as f:
        json.dump(sessions, f)
    print(f"{len(sessions)} sessions collected.")

    # Get docids mentioned
    docids = set()
    for qid, v in sessions.items():
        docids.update(v["positive_items"])
        docids.update(v["negative_items"])
    print(f"{len(docids)} docids found.")

    # Get attributes for these docids
    items_dict = {}
    with pd.read_csv(docs_file, header=None, sep="\t", chunksize=SAMPLE_ROWS) as reader:
        for docs in tqdm(reader):
            docs.columns = ["docid", "url", "title", "text"]
            docs = docs[docs["docid"].isin(docids)]
            for _, row in docs.iterrows():
                items_dict[row["docid"]] = " ".join(
                    (str(row["title"]) + " " + str(row["text"])).split()[:N_WORDS]
                )
    with open(items_file, "w") as f:
        json.dump(items_dict, f)


def get_msmarco_sample_data():
    package_path = "mini_rec_sys.sample_data.msmarco_reranking"
    with resources.open_text(package_path, items_file) as f:
        items = json.load(f)
    with resources.open_text(package_path, sessions_file) as f:
        sessions = json.load(f)
    return items, sessions


if __name__ == "__main__":
    generate_and_write_data()
