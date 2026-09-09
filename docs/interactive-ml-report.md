# Portable interactive ML report

Point 17 combines the already verified analysis views into one self-contained HTML document. It does not run
embeddings, dimensionality reduction, clustering, reassignment, or pipeline resume. Open
`notebooks/02_colab_results_analysis.ipynb`, set `WRITE_INTERACTIVE_REPORT=True`, and run the notebook in order.

The output is written outside the immutable run:

```text
/content/drive/MyDrive/b2b-radar/interactive-reports/<RUN_ID>/
├── interactive-report.html
└── interactive-report-manifest.json
```

The HTML embeds its CSS, JavaScript, and report data and therefore opens without a server or internet connection. It
contains an overview, scope-aware data flow, searchable topic cards, quality and problem-signal indicators, source
concentration, monthly dynamics, warnings, and metric interpretation notes. The manifest binds the document to the
pipeline-manifest checksum, configuration, analysis timestamp, and HTML checksum.

## Privacy modes

`INTERACTIVE_REPORT_PRIVATE_EXAMPLES=False` is the default. In this mode representative comments are never loaded,
the report is classified `aggregate_internal`, and only aggregate values are embedded.

Set `INTERACTIVE_REPORT_PRIVATE_EXAMPLES=True` only when the recipient is allowed to read source comments. The report
then includes up to `INTERACTIVE_REPORT_EXAMPLES_PER_TOPIC` persisted representatives and verified YouTube links and
is classified `restricted`. Author identifiers are not included. Store and share this file under the same access and
retention controls as the source dataset. Changing this switch requires regenerating the report but never invalidates
ML checkpoints.

Existing output is not overwritten unless `OVERWRITE_REPORT=True`. Generation fails on checksum, lineage, alignment,
or timezone errors rather than producing a plausible-looking report from inconsistent artifacts. Topic status,
priority, source concentration, and temporal movement remain diagnostics requiring manual interpretation; they are
not verified severity, causal demand, or business impact.
