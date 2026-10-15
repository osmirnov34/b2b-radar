# Local analysis data

This directory is intentionally excluded from Git except for this guide.

- `raw/` — immutable downloaded source files.
- `interim/` — reproducible development/validation/test splits and other intermediate files.
- `processed/` — cleaned data, embeddings, deduplication, corpus, UMAP, clusters, topics, and final outlier decisions.
- `samples/` — deterministic research subsets and metadata.
- `reports/` — dataset profiles and sanitized validation errors.

Run a privacy-safe structural audit of a downloaded Yandex Maps reviews export before adapting or processing it:

```bash
uv run python scripts/audit_yandex_reviews.py data/raw/yandex-reviews.jsonl \
  --report-dir data/reports/yandex-reviews \
  --minimum-reviews 30
```

The JSON and Markdown reports contain aggregate field, identifier, organization, rating, date, duplicate, and text
length statistics. They intentionally exclude review text and author values. The source file is read-only and is never
rewritten by the audit.

After a successful audit, create a source-neutral private analytical artifact at a new path:

```bash
uv run python scripts/prepare_yandex_feedback.py \
  data/raw/yandex-reviews.jsonl \
  data/interim/yandex-feedback.jsonl
```

The command also writes `data/interim/yandex-feedback.jsonl.manifest.json` with input/output checksums, record and
organization counts, rating distribution, and date coverage. It never replaces existing files: use a new output path
for a new dataset version or run. The canonical JSONL retains review text for downstream analysis but excludes author
display names. Treat both the JSONL and its manifest as private local artifacts and do not commit them.

Never commit comments, embeddings, generated reports, or model artifacts.
Topic keywords and generated names can contain sensitive terms derived from comments; review them before export.
