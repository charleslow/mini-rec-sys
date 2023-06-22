#!/bin/bash
# Note: These will take awhile to download and unzip
wget https://msmarco.blob.core.windows.net/msmarcoranking/orcas-doctrain-top100.gz
gunzip orcas-doctrain-top100.gz
mv orcas-doctrain-top100 orcas-doctrain-top100.tsv
wget https://msmarco.blob.core.windows.net/msmarcoranking/orcas-doctrain-queries.tsv.gz
gunzip orcas-doctrain-queries.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
gunzip msmarco-docs.tsv.gz