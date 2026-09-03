# Google Colab ML runbook

Open `notebooks/00_colab_pipeline.ipynb` in Google Colab and select a GPU runtime. The notebook clones the maintained
repository, records the resolved Git commit, installs the `analysis` dependency group, mounts Google Drive, and either
downloads the configured public Drive file or copies an explicitly mounted Drive path.

Python 3.11 through 3.13 are supported. The setup cell upgrades the packaging tools, installs the ML dependencies with
the active interpreter, and records their versions. After data validation, a small synthetic compatibility check runs
UMAP, HDBSCAN, and HNSWLIB before any corpus-scale work. A failure stops the notebook before smoke/full execution. The
other notebooks cloned with the repository are inert and are never executed automatically.

PyPI currently provides CPython 3.13 Linux wheels for the main numeric and ML stack but not for `hnswlib` 0.8.0. On
Python 3.13 the notebook therefore checks for `g++` and performs one explicit, pinned source build before the general
project installation. This exception is recorded as `controlled-source-build` in `colab-environment.json`; no other
source build is intentionally requested.

The notebook does not assume that the downloaded object is comments data. It detects the format and accepts only
UTF-8 JSONL whose records validate as `ExportedComment`. HTML permission pages, archives, JSON arrays, CSV, corrupt
records, and large count mismatches stop before any model is loaded. No automatic format conversion is performed.
The expected row count is approximate: `RECORD_COUNT_TOLERANCE` defaults to `0.25`, so the notebook accepts a symmetric
relative difference of up to 25%. It prints and records expected, actual, signed difference, relative difference,
tolerance, and the comparison result. This tolerance applies only to record count; format and schema checks stay strict.

## Execution gates

Both expensive flags default to false:

```python
RUN_SMOKE = False
RUN_FULL = False
```

Run all cells through dry-run first. Then enable smoke-run for a 2,000-record deterministic whole-video sample. Enable
a new full run only after smoke passes in the same session. Resume is allowed without repeating smoke after the source
checksum, configuration snapshots, stage markers, and replacement plan pass dry-run.

The notebook pins the embedding model revision and creates untracked runtime configurations under `/content`. Source
data and the Hugging Face cache use Colab's local disk. Run directories use mounted Drive so manifests and completed
stage artifacts survive runtime replacement. Direct Drive I/O must be proven by smoke-run before the full corpus.

## Shared-file access

The default file ID is the team-provided Drive object. If public download fails, add a shortcut to the file in My
Drive, set `DRIVE_INPUT_PATH` to its mounted path, and re-run only the acquisition and subsequent cells. Never commit
the downloaded comments or a notebook containing raw-record output.

## Resume

Keep `RUN_ID` unchanged, recreate the same local input path with identical bytes, and set:

```python
RUN_FULL = True
RESUME_RUN_DIR = "/content/drive/MyDrive/b2b-radar/ml-runs/<run-id>"
```

If dry-run identifies an interrupted stage, set `RESTART_FROM` only to the stage it reports. Publication is deliberately
absent from this notebook; manual evaluation and a reviewed production snapshot remain separate operations.

Smoke and full execution print versioned stage events and elapsed-time heartbeats. The same aggregate events are stored
under `<run-dir>/observability/`; see `docs/ml-progress.md`. After topic artifacts exist, use the independent read-only
`notebooks/02_colab_results_analysis.ipynb` described in `docs/ml-results-analysis.md` for charts and review tables.
