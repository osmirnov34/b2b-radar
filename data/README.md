# Local analysis data

This directory is intentionally excluded from Git except for this guide.

- `raw/` — immutable downloaded source files.
- `interim/` — reproducible development/validation/test splits and other intermediate files.
- `processed/` — cleaned data, embeddings, deduplication indexes, aligned corpus, UMAP, clusters, and local topics.
- `samples/` — deterministic research subsets and metadata.
- `reports/` — dataset profiles and sanitized validation errors.

Never commit comments, embeddings, generated reports, or model artifacts.
Topic keywords and generated names can contain sensitive terms derived from comments; review them before export.
