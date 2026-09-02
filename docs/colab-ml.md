# Google Colab ML runbook

Open `notebooks/00_colab_pipeline.ipynb` in Google Colab and select a GPU runtime. The notebook clones the maintained
repository, records the resolved Git commit, installs the `analysis` dependency group, mounts Google Drive, and either
downloads the configured public Drive file or copies an explicitly mounted Drive path.

The notebook does not assume that the downloaded object is comments data. It detects the format and accepts only
UTF-8 JSONL whose records validate as `ExportedComment`. HTML permission pages, archives, JSON arrays, CSV, corrupt
records, and large count mismatches stop before any model is loaded. No automatic format conversion is performed.

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
