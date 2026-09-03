# ML pipeline progress

Pipeline execution emits versioned, aggregate `ProgressEvent` records without comment text or source provenance. Every
run writes them under `ml-runs/<run-id>/observability/`; `progress-events.jsonl` is append-only and
`current-status.json` is replaced atomically. These files are observational: they are excluded from model configuration
hashes, stage markers, and resume compatibility.

The Colab pipeline enables live output for smoke and full execution. It reports stage start, verified completion,
failure, resume/skip decisions, and a 30-second elapsed-time heartbeat while a subprocess is active. A heartbeat means
only that the stage process is running; it is not a checkpoint or a percentage estimate. UMAP and HDBSCAN do not expose
a reliable internal completion percentage.

If console or journal reporting fails, the computation continues and prints only the exception type. A stage is called
complete only after its existing marker and checksum have been verified. Runs created before observability existed
remain resumable; the directory is created on the next resume without changing prior stage records.

Progress reporting and result visualization are deliberately independent. `ProgressEvent.schema_version` can evolve
without changing pipeline or report schemas, and the result notebook never uses progress events as model inputs.
