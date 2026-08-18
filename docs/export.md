# Stage 13: result export

Stage 13 converts checksum-bound outputs from stages 7–12 into stable JSONL contracts for application and research
consumers. It never changes labels and rejects broken artifact lineage, incomplete topic representations, invalid
arrays, and unknown topic IDs.

## Command

```bash
python scripts/export_topic_results.py \
  --corpus-manifest data/processed/corpus/corpus-manifest.json \
  --clustering-manifest data/processed/clustering/clustering-manifest.json \
  --topic-manifest data/processed/topics/topic-representation-manifest.json \
  --reassignment-manifest data/processed/outlier-reassignment/outlier-reassignment-manifest.json \
  --evaluation-manifest data/processed/evaluation/evaluation-manifest.json \
  --config configs/export.example.json \
  --output-dir data/processed/export
```

Set `require_final_evaluation` to reject preliminary stage-12 results. The default allows a clearly marked preliminary
research export so the pipeline can be inspected before validation is complete.

## Public contracts

- `topics.jsonl` contains names, filtered keywords, final sizes, and explicitly named original-cluster aggregates.
- `assignments.jsonl` maps the internal hashed `corpus_id` to a topic, confidence, and assignment source.
- `quality.json` propagates the stage-12 status, preliminary flag, warnings, and selected aggregate metrics.
- `export-manifest.json` records lineage, row counts, paths, SHA-256 hashes, configuration, and schema version.

Public files contain no text, source record ID, author, video metadata, or search query. Email- or phone-like keyword
terms are removed. Remaining outliers use `topic_id: null`, source `outlier`, and an explicit rejection reason.

## Sensitive research export

`include_research_text: true` additionally creates `research-assignments.jsonl` with text and provenance. This is an
explicitly sensitive local artifact: keep it below ignored `data/`, do not commit it, and do not serve it through the
web application. Existing output is protected unless `--force` is supplied; the manifest is published last.
