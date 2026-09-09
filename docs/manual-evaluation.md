# Manual ML evaluation

Open `notebooks/04_colab_manual_evaluation.ipynb` after stage 12 has created
`manual-review-sample.jsonl`. The pipeline may be in `awaiting_review`; final export is intentionally blocked until
the separate evaluation workflow is completed and approved.

Set `RUN_ID` and a stable, non-personal reviewer identifier. The notebook verifies the run manifest, full evaluation
manifest, sample checksum, and template checksum. If
`/content/drive/MyDrive/b2b-radar/manual-reviews/<run-id>/<reviewer>/manual-annotations.jsonl` exists, it is loaded and
validated so work can continue in a later Colab session.

Private text is hidden by default. Set `SHOW_PRIVATE_TEXT=True` only in a restricted session, then use:

```python
record_index = next_unreviewed_record_index()
show_review_item(record_index)
set_annotation(
    record_index,
    topic_matches=True,
    topic_clear=True,
    business_relevant=True,
    reassignment_correct=None,
)
```

Every annotation records whether the assigned topic matches, whether the topic is clear, and whether it is relevant
to the product. A reassigned sample additionally requires `reassignment_correct`; that field must remain `None` for
original members and remaining outliers. Reviewers may flag sensitive data and merge/split candidates. Notes are
limited to 2,000 characters, reviewer identity is mandatory, record indices must belong to the verified sample, and
duplicates are rejected.

`summarize_manual_annotations()` reports completion and outcome shares. These are reviewer judgments, not ground
truth. Optional independently produced annotation files can be compared with `reviewer_agreement()`. Its result is
raw observed agreement over overlapping binary judgments, not a chance-corrected reliability coefficient.

Nothing is saved until `SAVE_ANNOTATIONS=True`. Saving writes an atomic pipeline-compatible JSONL file and a checksum
manifest outside `ml-runs`, alongside the experiment passport. The directory is classified
`restricted_manual_review`: annotations and notes may reveal conclusions about private source text and must not be
placed in the aggregate report or public export.

To use the completed file in stage 12, copy the evaluation configuration, set `validation_completed=true`, configure
`manual_annotations` to the saved JSONL path, run dry-run against the same `RUN_ID` and checksums, then explicitly
resume from `evaluation`. Do not mark validation complete solely because the notebook reached 100% coverage; review
quality and disagreement first.
