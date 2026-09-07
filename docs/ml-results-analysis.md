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

## Cluster cards

Set `SELECTED_TOPIC_ID` to inspect one normalized topic. `build_cluster_cards()` reads only checksum-verified aggregate
cluster, topic, and optional reassignment summaries; it does not load raw comments or author identifiers. The card
shows its name and keywords, original and final size, corpus share, comment/reply and language composition, unique
videos, representative count, pipeline status, source manifest rows, and any reassignment exclusion reason.

The card deliberately keeps `original_*_probability` separate from `*_reassignment_similarity`. The first values are
HDBSCAN membership probabilities for original members; the second values are embedding cosine similarities for
outliers accepted during reassignment. They are different measurements and must not be ranked as one confidence
scale. If a topic is excluded from reassignment, the card reports the persisted reason and does not imply that its
outliers were considered for assignment.

## Representative comments

Representative text remains hidden until `SHOW_PRIVATE_TEXT=True`. The notebook function
`show_representative_comments(artifacts, n=None, topic_id=None)` displays the representatives persisted by stage 10,
their centroid-similarity rank, original HDBSCAN membership probability, text role, video metadata, and a validated
YouTube source link. Author identity is deliberately omitted.

`n` is not tied to a reporting constant. `None` returns every representative available in the selected source run; a
positive integer limits each topic independently. Asking for more records than were persisted returns all available
records without failure or recomputation. For example, a run produced with `representatives_per_topic=5` can display
up to five representatives per topic, while a future run that persists more works without changing the notebook.

The function verifies the representative file checksum, its agreement with each topic representation, corpus index
bounds, original HDBSCAN label membership, label/probability alignment, and final corpus row count. It reads the
existing corpus and never runs embeddings, UMAP, HDBSCAN, topic construction, or reassignment.

## Assignment review

After opting in with `SHOW_PRIVATE_TEXT=True`,
`show_assignment_review_comments(artifacts, n=10, topic_id=None)` exposes up to `n` questionable records per topic
and review kind. It provides three separate review queues:

- original cluster members with the lowest HDBSCAN membership probability;
- accepted outlier reassignments with the smallest cosine-similarity margin;
- rejected outliers closest to a candidate topic, together with the persisted rejection reason.

An accepted reassignment has an `assigned_topic_id`. For a rejected outlier this field remains empty: its
`candidate_topic_id` is only the nearest topic considered by stage 11 and must not be presented as the record's
class. HDBSCAN probability, best and second centroid cosine similarity, and similarity margin remain separate
columns because they are not interchangeable confidence scores.

The function verifies original assignment checksums, the optional reassignment-decision checksum, decision count,
unique and in-range indices, original-outlier status, and final-label consistency. Runs without stage 11 can still
show low-probability original members. The report reads existing artifacts only, omits authors, validates source
links, and performs no embeddings or clustering work.

When `SAVE_REPORT=True`, output is written outside the immutable run tree to
`/content/drive/MyDrive/b2b-radar/visualizations/<run-id>/`. Existing reports are not overwritten unless
`OVERWRITE_REPORT=True`. The report manifest has its own schema version and records the source pipeline-manifest hash;
changes to colors, charts, or report dependencies therefore cannot invalidate ML checkpoints or resume state.
