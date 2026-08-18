# Web subsystem

FastAPI routes, templates, and UI orchestration live here. The web layer may start ingestion and import final ML
exports, but it must not implement preprocessing, embeddings, or clustering.

The optional topic API consumes only the public stage-13 export through `ML_EXPORT_MANIFEST`. It validates strict
schemas, hashes, counts, normalized topic ownership, and local artifact paths. It never opens the research export.
Results that are preliminary or failed remain unavailable unless `ML_ALLOW_UNRELIABLE=true` is explicitly set for a
development environment.

Available read-only routes:

- `GET /api/topics`
- `GET /api/topics/{topic_id}`
- `GET /api/topics/{topic_id}/assignments`
- `GET /api/ml/outliers`
- `GET /api/ml/quality`

The repository reloads only after the manifest changes. A broken update records an error but keeps serving the last
valid snapshot. Deploy artifact files first and publish `export-manifest.json` last.
