# User topic and industry search

Point 18 adds a first product-oriented retrieval layer over an existing, checksum-verified ML run. A user enters an
industry, product area, or theme; the system embeds that query with the exact model name, revision, prompt, and vector
dimension recorded by stage 5, then compares it with the retained stage-7 corpus embeddings. It does not retrain UMAP
or HDBSCAN and does not change cluster assignments or checkpoints.

Open `notebooks/05_colab_topic_search.ipynb`, select the source `RUN_ID`, edit `USER_TOPIC`, review the thresholds, and
set `RUN_SEARCH=True`. A GPU is recommended for loading the multilingual embedding model, while similarity scanning is
batched over the memory-mapped embedding matrix.

The result contains:

- total relevant and problem-evidence rows;
- matching persisted topics and optional outliers;
- semantic similarity and the separate persisted cluster confidence;
- lexical problem markers using the same negation policy as aggregate reporting;
- bounded evidence examples with comment/reply role, date, video metadata, and a validated YouTube URL;
- source run, corpus, embedding, model, revision, configuration, and timestamp provenance.

No author identifier is returned. Evidence still contains source text and links, so the complete result is classified
`restricted_user_topic_search`. Display requires `SHOW_RESTRICTED_EVIDENCE=True`; persistence independently requires
`SAVE_RESTRICTED_RESULT=True`. Saved output goes to:

```text
/content/drive/MyDrive/b2b-radar/topic-searches/<RUN_ID>/<QUERY_HASH>/
├── topic-search-result.json
└── topic-search-manifest.json
```

Existing output is not overwritten unless `OVERWRITE_RESULT=True`. The manifest stores a query hash rather than a
second plaintext query copy and binds the result to the pipeline and result checksums.

## Interpretation limits

This is retrieval and evidence triage—not a factual verdict that a problem exists, a severity score, or an estimate of
industry prevalence. `minimum_similarity` needs empirical calibration on representative queries; a universal value is
not implied. The searchable corpus is the retained development corpus and inherits collection queries, source-video
selection, language filtering, cleaning, and deduplication biases. High semantic similarity, HDBSCAN confidence, and
problem markers are separate measurements and must not be treated as interchangeable probabilities.

The current layer reuses the fixed corpus and topics. A future product can add authenticated web/API access, a reviewed
query benchmark, calibrated thresholds, access logging, retention enforcement, and analyst feedback without changing
the retrieval contract.
