# Internal analysis models

Offline analysis keeps persisted JSON contracts separate from strict internal value objects:

```text
ExportedComment
      |
      | normalization
      v
CommentRecord
      |
      | cleaning -> deduplication -> topic assignment
      v
AnalysisResult
      |
      | reporting
      v
ClusterRecord + AnalysisRunMetadata
```

Transport contracts in `src/analysis/schemas.py` accept unknown additive fields for forward compatibility. Internal
models in `src/analysis/models.py` are frozen and reject unknown fields so programming and notebook mistakes fail close
to their source.

## Comments

`CommentRecord` is the normalized unit passed between analysis stages. `from_export()` maps the web JSONL names to the
internal names and trims the comment text. Missing provenance is already normalized by `ExportedComment`.

`normalized_text_key` uses Unicode-aware `casefold()` and collapses whitespace for exact duplicate detection without
changing the original text. `video_url` preserves an exported source URL when present and otherwise constructs the
standard YouTube watch URL from `video_id`. `to_cluster_comment()` is the single conversion back to the persisted
cluster-comment contract.

## Stage results

- `CleaningDecision`, `CleaningStats`, and `CleaningResult` explain which comments were kept or removed.
- `DuplicatePair`, `DuplicateGroup`, `DeduplicationStats`, and `DeduplicationResult` describe semantic deduplication.
- `TopicAssignment` and `TopicSummary` represent clustering without depending on BERTopic or pandas objects.
- `AnalysisCounts` validates the processing-count sequence used by run metadata.
- `AnalysisResult` is the future pipeline boundary combining configuration, normalized comments, topics, output
  contracts, and reproducibility metadata.

The internal models deliberately contain no NumPy arrays, Torch tensors, pandas frames, or BERTopic instances. Those
runtime objects have separate lifecycles and will be represented by stage-specific services in later refactoring.
