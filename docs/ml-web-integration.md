# Stage 14: web integration of topic results

The FastAPI application consumes the immutable public snapshot created by stage 13. It does not import preprocessing,
embedding, clustering, or research-export code. The integration is optional, read-only, and independent from normal
ingestion and database routes.

## Configuration

```dotenv
ML_RESULTS_ENABLED=true
ML_EXPORT_MANIFEST=data/processed/export/export-manifest.json
ML_ALLOW_UNRELIABLE=false
```

With ML disabled or no valid snapshot available, the rest of the application starts normally and ML endpoints return
HTTP 503. `ML_ALLOW_UNRELIABLE=true` is intended only for development inspection: otherwise preliminary and failed
stage-12 evaluations are rejected.

## API

- `GET /api/topics` lists aggregate public topics.
- `GET /api/topics/{topic_id}` returns one topic or HTTP 404.
- `GET /api/topics/{topic_id}/assignments?offset=0&limit=100` returns hashed corpus assignments.
- `GET /api/ml/outliers?offset=0&limit=100` returns records with no topic and their reason.
- `GET /api/ml/quality` returns propagated evaluation quality and a safe hot-reload warning.

Assignment pages are limited to 500 rows. They contain `corpus_id`, final topic, confidence, source, and optional
outlier reason; there is no source ID, text, author, query, or video metadata.

## Publication and failure behavior

Deploy `topics.jsonl`, `assignments.jsonl`, and `quality.json` first, then atomically publish `export-manifest.json`.
The reader accepts only these exact filenames beside the manifest, checks their hashes and strict schemas, verifies
counts and topic ownership, and refuses any manifest that references `research-assignments.jsonl`.

The repository checks for a changed manifest on access. If a replacement is incomplete or invalid, it keeps serving
the last known good immutable snapshot and exposes only a generic reload warning. Invalid record content and local
filesystem paths are never returned by the API.
