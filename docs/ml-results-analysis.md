# ML result analysis

Open `notebooks/02_colab_results_analysis.ipynb` after clustering and topic representation have completed. Set `RUN_ID`
to the persisted run under `/content/drive/MyDrive/b2b-radar/ml-runs/`. A GPU is not required. The notebook validates
the source run and checksums before loading aligned coordinates, labels, confidence, topic representations, and the
optional outlier-reassignment output.

The default is strictly read-only:

```python
SAVE_REPORT = False
SHOW_PRIVATE_TEXT = False
```

It displays aggregate topic tables, cluster sizes, outlier share, assignment-confidence distribution, a deterministic
stratified UMAP view, keyword-based topic similarity, and source concentration by language, query, and video. Private
representative text is shown only after an explicit local opt-in and is never included in generated aggregate tables.

When `SAVE_REPORT=True`, output is written outside the immutable run tree to
`/content/drive/MyDrive/b2b-radar/visualizations/<run-id>/`. Existing reports are not overwritten unless
`OVERWRITE_REPORT=True`. The report manifest has its own schema version and records the source pipeline-manifest hash;
changes to colors, charts, or report dependencies therefore cannot invalidate ML checkpoints or resume state.
