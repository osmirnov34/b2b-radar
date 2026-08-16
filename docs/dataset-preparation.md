# Dataset preparation

The full comment corpus is local data and must never be committed. Place the unchanged download at
`data/raw/comments.jsonl`; derived samples and reports stay under `data/` and are ignored by Git.

## 1. Download and provenance

Download the shared Google Drive file in a browser and preserve its original name until its physical format is known:

```bash
mkdir -p data/raw data/interim data/processed data/samples data/reports
file data/raw/<downloaded-file>
file --mime data/raw/<downloaded-file>
du -h data/raw/<downloaded-file>
sha256sum data/raw/<downloaded-file>
```

If it is confirmed as uncompressed UTF-8 JSONL, store it as `data/raw/comments.jsonl`. Record the source URL,
download time, original filename, byte size, checksum, and approximate expected count in a local
`data/raw/source.json`. Do not commit that sidecar.

Example local provenance:

```json
{
  "source_url": "https://drive.google.com/file/d/.../view",
  "downloaded_at": "2026-08-16T12:00:00+03:00",
  "original_filename": "comments.jsonl",
  "size_bytes": 0,
  "sha256": "...",
  "expected_records": 260000
}
```

## 2. Inspect the complete file

The inspector reads JSONL line by line and does not load the full corpus into a DataFrame. It computes the checksum,
validates `ExportedComment`, profiles lengths and provenance, counts exact text/ID duplicates, and stores only sanitized
validation errors. Original comment text and field values are never written to error reports.

```bash
uv run python scripts/inspect_dataset.py data/raw/comments.jsonl \
  --expected-records 260000 \
  --expected-records-tolerance 0.10 \
  --report-dir data/reports
```

Generated files:

```text
data/reports/dataset-profile.json
data/reports/dataset-profile.md
data/reports/inspection-errors.jsonl
```

The physical detector distinguishes JSONL, JSON arrays/objects, CSV, TSV, HTML, ZIP, GZIP, binary, and empty files.
This catches the common case where a Google Drive confirmation or authorization HTML page is accidentally saved with
a `.jsonl` extension.

Example failure:

```text
Expected format: jsonl
Detected format: html
ERROR: expected jsonl, detected html
```

Exit codes:

- `0` — format, schema, expected size, and error threshold are acceptable;
- `1` — the file was inspected but is not safe to use;
- `2` — missing file, invalid arguments, or invalid thresholds.

## 3. Create a research sample

A deterministic sample can be produced during the same inspection:

```bash
uv run python scripts/inspect_dataset.py data/raw/comments.jsonl \
  --expected-records 260000 \
  --sample-size 10000 \
  --sample-output data/samples/comments-10k.jsonl \
  --sample-seed 42 \
  --max-per-video 100
```

Sampling uses a stable hash rank and a per-video cap so a single popular video cannot dominate the notebook dataset.
The adjacent `.meta.json` records source/output checksums, seed, requested/written counts, and the cap. Invalid or empty
records are skipped. If the cap prevents reaching the requested count, `written_records` explicitly reports the smaller
result.

## 4. Split the corpus

Run this only after the complete file passes inspection. The default split is 70% development, 15% validation, and
15% test. A stable SHA-256 assignment keeps every `video_id` in exactly one partition and produces the same result for
the same seed regardless of input row order. When `video_id` is absent, the fallback order is video URL,
channel + title, comment ID, then a hash of the canonical record.

```bash
uv run python scripts/split_dataset.py data/raw/comments.jsonl \
  --output-dir data/interim/splits \
  --development-ratio 0.70 \
  --validation-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42
```

Generated files:

```text
data/interim/splits/development.jsonl
data/interim/splits/validation.jsonl
data/interim/splits/test.jsonl
data/interim/splits/split-manifest.json
data/interim/splits/split-report.md
```

The split is group-safe first. It then audits normalized exact text with ownership priority
`development > validation > test`. A sufficiently informative duplicate already owned by a higher-priority partition
is omitted from the lower-priority output. Short generic phrases such as acknowledgements are retained and counted as
ignored noise overlaps; they never connect unrelated videos. Semantic near-duplicates are deliberately deferred until
the embedding/ANN stage.

The manifest contains checksums, hashed group assignments, parameters, counts, and leakage statistics. It contains no
comment text or raw video identifiers. The command refuses to overwrite an existing split unless `--force` is passed.
Large deviations from the target ratios are shown in `split-report.md`; this can occur when a few videos dominate the
record count and must be reviewed before model evaluation.

Notebook exploration and tuning must use only `development.jsonl`. Use `validation.jsonl` for model/parameter
selection and open `test.jsonl` only for the final evaluation.

## 5. Run development EDA

The EDA command verifies the development checksum and record count against the split manifest before profiling. All
corpus-wide metrics are streamed; Lingua runs on a deterministic bounded sample because full language detection is the
expensive part. No language is filtered at this stage.

```bash
uv run python scripts/run_eda.py \
  data/interim/splits/development.jsonl \
  --manifest data/interim/splits/split-manifest.json \
  --report-dir data/reports/eda \
  --language-sample-size 20000 \
  --language-seed 42
```

Add a local, development-only notebook sample when needed:

```bash
uv run python scripts/run_eda.py \
  data/interim/splits/development.jsonl \
  --manifest data/interim/splits/split-manifest.json \
  --report-dir data/reports/eda \
  --sample-size 20000 \
  --sample-output data/samples/development-eda.jsonl \
  --sample-seed 42 \
  --max-per-video 100
```

Outputs include `development-profile.json`, a Markdown summary, aggregate CSV tables, and SVG charts for text length,
duplicate groups, language estimates, time, and noise signals. Group labels are hashed. Raw texts, authors, channel
names, queries, and video IDs are never written to EDA reports. The optional local sample does contain comments and is
therefore ignored by Git.

Open `notebooks/01_development_eda.ipynb` for a thin presentation of the safe profile. The notebook intentionally does
not display raw comments, validation, or test data. Re-run the CLI after changing the source split; checksum validation
prevents stale or substituted data from being analysed silently.

## 6. Blocking conditions

The dataset is blocked when:

- its physical format is not UTF-8 JSONL;
- no rows match `ExportedComment`;
- the row error rate exceeds `--max-error-rate` (default 1%);
- the valid record count differs from `--expected-records` beyond the configured tolerance.

Missing authors, dates, queries, duplicates, and individual empty comments are reported but are not format-level
failures. Cleaning decisions happen in a later pipeline stage; inspection never mutates `data/raw`.
