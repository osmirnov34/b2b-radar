# Stage 7: final aligned corpus

Stage 7 applies the checksum-verified keep indexes from semantic deduplication to both cleaned records and their
multilingual embeddings. It is the single input boundary for UMAP and later clustering stages.

## Command

```bash
uv run python scripts/build_corpus.py \
  data/processed/cleaning/development-clean.jsonl \
  data/processed/embeddings/embeddings.npy \
  data/processed/semantic-deduplication/keep-indices.json \
  --cleaning-manifest data/processed/cleaning/cleaning-manifest.json \
  --embedding-manifest data/processed/embeddings/embedding-manifest.json \
  --groups data/processed/semantic-deduplication/semantic-groups.jsonl \
  --deduplication-manifest data/processed/semantic-deduplication/semantic-deduplication-manifest.json \
  --output-dir data/processed/corpus
```

Use `--force` only to replace a completed corpus. Existing outputs are protected by default.

## Output contract

```text
final-corpus.jsonl
final-embeddings.npy
final-record-ids.jsonl
corpus-manifest.json
corpus-report.md
```

For every row `i`:

```text
final-corpus[i].record_id == final-record-ids[i]
final-corpus[i] <-> final-embeddings[i]
```

Each corpus record preserves the cleaned unit and adds `corpus_id`, `cleaned_record_index`, and
`semantic_duplicate_count`. The stable corpus ID is a SHA-256-derived identifier; the cleaned index links back to
the stage 4 record without putting raw text into audit reports. The semantic groups from stage 6 remain the detailed
mapping for excluded records.

## Validation and publication

The builder verifies cleaning, embedding, record-ID, keep-index, group, and semantic-deduplication checksums. Keep
indexes must be unique, increasing, and in range. Groups must account for exactly every excluded index, have retained
representatives, and never overlap. The source matrix must be aligned `float32` with its manifest.

JSONL is streamed and the selected matrix is written to a NumPy memmap in chunks. All output files are built under
temporary names; the manifest is published last and acts as the completion marker. Reports contain aggregate values
and checksums only.

## Loading for stage 8

```python
import json
import numpy as np

manifest = json.loads(open("data/processed/corpus/corpus-manifest.json", encoding="utf-8").read())
embeddings = np.load("data/processed/corpus/final-embeddings.npy", mmap_mode="r", allow_pickle=False)
assert embeddings.shape == (manifest["stats"]["output_records"], manifest["dimensions"])
```
