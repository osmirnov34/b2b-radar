# Offline ML subsystem

This package owns reproducible, offline processing after comments have been exported. Its input boundary is versioned
JSONL plus manifests; it must not call YouTube, open database sessions, import FastAPI, or depend on ingestion
implementations.

Current stages cover contracts, dataset inspection and splitting, EDA, normalization, cleaning, multilingual
embeddings, semantic deduplication, final-corpus assembly, separate UMAP spaces, and checksum-bound HDBSCAN
clustering, followed by fixed-label c-TF-IDF topics and conservative embedding-based outlier reassignment. Evaluation
adds geometry, bootstrap stability, reassignment comparison, and a human-review gate. A checksum-bound export layer
then publishes safe application contracts and an explicit local research variant. Large
vectors and generated data stay under ignored `data/`; persisted metadata uses typed manifests and SHA-256 checksums.

Dependency direction:

```text
domain       <- ingestion
domain       <- web
ml           -> text_processing
ingestion    -> text_processing
ml           -X-> ingestion, web, database infrastructure
```

CLI files under `scripts/` are thin adapters. Notebooks may read safe aggregates or call this package, but must not
contain the only implementation of an ML stage.
