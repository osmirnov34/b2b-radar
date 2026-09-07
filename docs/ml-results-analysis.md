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

## Auditable data flow

The first report section uses `build_data_lineage()` instead of treating every count as one linear funnel. Every
metric carries its dataset scope (`all`, `development`, `validation`, or `test`), entity type, source manifest, exact
source field, and a plain-language definition. The notebook displays the leakage-safe parent-comment split separately
from the development-only processing flow.

`Valid parent comments` comes from `01-inspection/dataset-profile.json:contract_valid` and covers the complete input.
`Flattened development text units` comes from
`04-cleaning/cleaning-manifest.json:stats.input_text_units` and equals development parent comments plus their nested
replies. It is therefore not compared directly with the complete-input count. Cleaning removal reasons and semantic
near-duplicate removals are displayed separately.

Before rendering, the report verifies input checksums and count invariants across inspection, split, cleaning,
semantic deduplication, corpus, final labels, and optional outlier reassignment. A missing, substituted, or
inconsistent artifact blocks the report with the failed invariant name instead of drawing a misleading chart.

When `SAVE_REPORT=True`, output is written outside the immutable run tree to
`/content/drive/MyDrive/b2b-radar/visualizations/<run-id>/`. Existing reports are not overwritten unless
`OVERWRITE_REPORT=True`. The report manifest has its own schema version and records the source pipeline-manifest hash;
changes to colors, charts, or report dependencies therefore cannot invalidate ML checkpoints or resume state.
