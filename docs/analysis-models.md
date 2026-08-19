# ML model boundaries

The maintained pipeline separates forward-compatible persisted contracts from strict internal stage models.

```text
ExportedComment
      ↓
TextUnit / CleanedTextUnit
      ↓
embedding, corpus, UMAP, HDBSCAN and topic artifacts
      ↓
stage-specific checksum manifests
      ↓
ExportedTopic / ExportedAssignment / ExportManifest
```

Transport schemas in `src/ml/schemas.py` tolerate additive fields where readers require forward compatibility.
Configuration, manifests, and internal value objects are frozen and reject unknown fields so invalid experiments fail
close to their source. Large arrays are stored as local artifacts rather than embedded in Pydantic objects.

`CommentRecord`, cleaning decisions, and semantic duplicate groups remain shared internal helpers used by maintained
stage implementations. The former all-in-one `AnalysisResult`, topic-assignment summary, and monolithic configuration
were removed with the obsolete pipeline.

`ClusterRecord` and `AnalysisRunMetadata` remain compatibility-only contracts. They allow the web application and its
tests to read historical `clusters.jsonl` uploads; they are not produced by the current pipeline. New application
integration consumes the strict stage-13 snapshot described in `ml-web-integration.md`.
