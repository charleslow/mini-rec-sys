Sample dataset curated from MS-Marco passage reranking task. https://microsoft.github.io/msmarco/.

## Task

Each session comprises a query and 100 candidate documents. The task is to rerank the documents by relevance.

## How the Dataset is created

First 50 queries are taken from https://msmarco.blob.core.windows.net/msmarcoranking/docleaderboard-queries.tsv.gz, and the corresponding documents from the top 100 candidates are retrieved to form the collection.

## Files

- `download.sh`: Download relevant files from msmarco
- `generate_data.py`: Clean up files to generate msmarco_sample.json