# Ingestion subsystem

This package discovers sources, calls external APIs, extracts comments/replies, applies inexpensive collection-time
gates, and persists documents. It owns network and database orchestration, but does not own embeddings, semantic
deduplication, clustering, or model evaluation.

The stable boundary to offline ML is the web JSONL export. Changes to exported text fields must be reflected in the
contracts under `src/ml` and covered by compatibility tests.
