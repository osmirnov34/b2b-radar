# ML experiment passport

`build_experiment_passport()` creates a compact reproducibility record for an existing pipeline run without copying
the dataset, configuration contents, comments, authors, source paths, or logs.

The passport records:

- run ID and pipeline status;
- source-dataset, configuration-snapshot, and run-manifest checksums;
- requested code ref, resolved Git commit, and dirty-worktree state;
- Python implementation/version and portable OS/machine class;
- installed versions of the configured ML/reporting packages;
- relative paths and checksums for every stage manifest;
- canonical checksums of embedded stage configurations rather than their values;
- verified stage marker/log checksums, return codes, durations, and completed-stage count.

Before construction, the configuration snapshot, every completed marker, every stage log, and all discovered
manifests must stay inside the selected run and must not be symbolic links. A checksum mismatch blocks passport
creation. Missing optional packages are recorded as `null`; failure to inspect Git produces `git_commit="unknown"`
and `git_dirty=null` rather than inventing provenance.

`write_experiment_passport()` writes `experiment-passport.json` atomically outside the immutable run. The manual
evaluation notebook stores it beside its restricted annotations. A passport describes provenance; it does not prove
model quality, approve manual validation, publish results, or make a dirty/unknown code state reproducible. Reviewed
runs should therefore use a commit SHA as `CODE_REF` and require `git_dirty=false` before external release.
