# Stage 10: fixed-label topic representation

Stage 10 builds c-TF-IDF keywords and representative row indexes from normalized HDBSCAN labels. It does not call
BERTopic `fit_transform`, UMAP, or HDBSCAN, so cluster membership and label `-1` remain unchanged. Outlier reassignment
belongs exclusively to stage 11.

## Command

```bash
uv sync --extra analysis
uv run python scripts/build_topic_representations.py \
  data/processed/corpus/final-corpus.jsonl \
  data/processed/clustering/cluster-labels.npy \
  --embeddings data/processed/corpus/final-embeddings.npy \
  --probabilities data/processed/clustering/cluster-probabilities.npy \
  --corpus-manifest data/processed/corpus/corpus-manifest.json \
  --clustering-manifest data/processed/clustering/clustering-manifest.json \
  --config configs/topic-representation.example.json \
  --output-dir data/processed/topics
```

`--limit-topics N` represents only the first N normalized topics for a disposable development trial. It never edits
the source labels; the manifest records source, represented, and omitted topic counts. Existing outputs require
`--force`.

## Representation pipeline

Cleaned texts are aggregated in memory by existing cluster ID. Outliers are skipped. The production backend uses
`CountVectorizer` followed by BERTopic's `ClassTfidfTransformer`, with configurable multilingual stopwords, n-grams,
BM25 weighting, and frequent-word reduction. A large cluster document can be capped by character count; truncation is
reported explicitly.

The top weighted terms form a deterministic working name such as `crm / integration / automation`. This is a machine
generated research label, not a final business interpretation.

Representative rows are ranked by mean HDBSCAN probability and cosine similarity to the cluster centroid. Selection
prefers different videos and authors, then fills remaining slots by score. Only row indexes and similarity values are
persisted. Raw author and video values are used transiently and are never written to reports.

## Artifacts

```text
topic-representations.jsonl
topic-keywords.jsonl
representative-indices.jsonl
vocabulary.json
ctfidf.npz
vectorizer.pkl
topic-representation-manifest.json
topic-representation-report.md
```

The manifest binds all corpus and clustering inputs, the stopword-set checksum, backend version, configuration, and
every generated artifact. Temporary files are published only after validation, with the manifest last.

The vectorizer is a local trusted pickle. This module writes it but provides no loader. Never deserialize an untrusted
file or skip checksum and origin verification.

Keywords and generated names are derived from local comments and may contain rare names or business terms. They stay
under ignored `data/` paths and must be reviewed for sensitive information before any external export.

## Quality diagnostics

The report records empty topics, vocabulary size, topic diversity, maximum keyword Jaccard overlap, similar-topic
pairs, duplicate generated names, representative similarity, and truncated cluster documents. It contains no topic
terms or comment values. Detailed terms remain only in the local topic artifacts.
