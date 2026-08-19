# Stage 15: reproducible ML operations

The operational runner connects stages 1–13 without moving their implementation into a notebook or one monolithic
process. Every stage remains a separate CLI and publishes its own checksum-bound manifest. The runner adds ordered
execution, preflight checks, local logs, resumability, a run manifest, the manual-review pause, atomic publication, and
rollback.

## Start a run

Copy `configs/pipeline.example.json`, set `source_dataset`, review all referenced versioned configurations, and run
from the repository root:

```bash
python scripts/run_ml_pipeline.py run configs/pipeline.example.json
```

Each run is written to `data/ml-runs/<run_id>/`. Set an explicit `run_id` for external scheduling, or leave it null for
a UTC timestamp. Preflight requires Python 3.11/3.12, all inputs/configurations/scripts, and the configured amount of
free disk space. No subprocess is launched through a shell.

For a deliberate partial run:

```bash
python scripts/run_ml_pipeline.py run configs/pipeline.example.json --stop-after corpus
```

Resume the same immutable inputs with:

```bash
python scripts/run_ml_pipeline.py run configs/pipeline.example.json \
  --run-dir data/ml-runs/<run_id> --resume
```

Resume verifies the source checksum, previous configuration snapshot, and checksum of every completed stage marker.
A changed or missing marker blocks continuation.

## Manual evaluation gate

The first evaluation normally ends with `awaiting_review` after creating `manual-review-sample.jsonl`. Complete the
local annotations, set `manual_annotations` in the pipeline configuration, and use an evaluation config with
`validation_completed: true`. Then explicitly rerun only evaluation and export:

```bash
python scripts/run_ml_pipeline.py run configs/pipeline.production.json \
  --run-dir data/ml-runs/<run_id> --resume --restart-from evaluation
```

`--restart-from` supplies `--force` only to that stage and later stages. The previous configuration snapshot remains
in the run directory and the resumed configuration receives a new filename. Export is never executed when evaluation
is preliminary or does not have status `pass`.

## Run state and recovery

`run-manifest.json` is atomically updated after every stage and contains:

- source and configuration hashes;
- ordered stage status, command, duration, return code, and finish time;
- marker and local-log hashes;
- overall `running`, `partial`, `awaiting_review`, `failed`, or `completed` status.

Logs are local ignored artifacts and should be treated as internal diagnostics. The manifest contains no comment text.
On failure, inspect the named log, correct the configuration or environment, and resume with `--restart-from` at the
failed stage. A partial export is never published.

## Publication and rollback

Publish only a completed stage-13 directory:

```bash
python scripts/run_ml_pipeline.py publish \
  data/ml-runs/<run_id>/13-export data/ml-published <run_id>
```

Publication first performs the same strict web validation: status must be final `pass`, all public files and hashes
must match, and no research export may be referenced. It then atomically switches `data/ml-published/current` and
keeps `previous` for rollback. Configure web with:

```dotenv
ML_RESULTS_ENABLED=true
ML_EXPORT_MANIFEST=data/ml-published/current/export-manifest.json
ML_ALLOW_UNRELIABLE=false
```

Rollback does not delete or rewrite a release:

```bash
python scripts/run_ml_pipeline.py rollback data/ml-published
```

## Retention policy

No automatic deletion is performed. Recommended policy:

- retain permanently: run manifest, configuration snapshots, stage manifests, final evaluation, and published export;
- archive when storage is constrained: embeddings, UMAP coordinates/models, HDBSCAN model, and topic matrices;
- retain locally with restricted access: manual samples, annotations, logs, and any research export;
- delete only through a separately reviewed operational procedure after confirming the run is not `current` or
  `previous` and archived checksums are recoverable.

## CI and smoke checks

The full production run needs the analysis dependencies, model weights, and real data, so CI uses the synthetic
orchestration smoke test plus the unit tests of every individual stage:

```bash
ruff check .
mypy src app.py
pytest -q
python scripts/run_ml_pipeline.py --help
pytest tests/test_ml_pipeline.py -q
```

The smoke test exercises ordered execution, checkpoint/resume, the manual-review stop, strict publication, and
rollback without network access or heavyweight model training.

## Read-only dry-run API (foundation)

The first dry-run implementation layer is available for notebook integration:

```python
from pathlib import Path

from src.operations import PipelineConfig, dry_run_pipeline

config_path = Path("configs/pipeline.example.json")
config = PipelineConfig.model_validate_json(config_path.read_text())
report = dry_run_pipeline(config, Path.cwd(), config_path=config_path)
```

This API creates no directory, manifest, log, temporary file, or symlink and launches no subprocess. It uses the same
pipeline context and command generator as a real run. It checks Python, scripts, input/config readability, analysis
packages, free disk space and permissions; streams the complete dataset for SHA-256 and line count; and validates the
first 100 non-empty records against the comment/reply contract without retaining or returning text values. The report
contains typed checks, stage actions, expected markers, a manual-gate forecast, and the command for a future real run.

CLI rendering, deeper typed configuration checks, publication dry-run, and the notebook display belong to the next
dry-run implementation increments; they are intentionally not duplicated inside this foundational API.
