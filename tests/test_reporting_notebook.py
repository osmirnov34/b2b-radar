import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/02_colab_results_analysis.ipynb"


def _sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]


def test_reporting_notebook_is_clean_and_parses() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    for source in _sources():
        ast.parse(source)
    for cell in notebook["cells"]:
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []


def test_reporting_notebook_is_read_only_by_default_and_separate_from_pipeline() -> None:
    source = "\n".join(_sources())

    assert "SAVE_REPORT = False" in source
    assert "SHOW_PRIVATE_TEXT = False" in source
    assert 'report_root = Path(DRIVE_PROJECT_DIR) / "visualizations"' in source
    assert "run_pipeline(" not in source
    assert "run_smoke_pipeline(" not in source
    assert "force=True" not in source
    assert "rm -rf" not in source
    assert "reset --hard" not in source


def test_reporting_notebook_covers_core_quality_views() -> None:
    source = "\n".join(_sources())

    assert "load_analysis_artifacts(run_dir)" in source
    assert "stratified_plot_indices" in source
    assert "Outlier share" in source
    assert "Assignment confidence" in source
    assert "build_data_lineage(artifacts)" in source
    assert "Leakage-safe parent-comment splits" in source
    assert "Development text flow" in source
    assert "source_field" in source
    assert "Text unit:" in source
    assert "SELECTED_TOPIC_ID = 0" in source
    assert "build_cluster_cards(artifacts)" in source
    assert "get_cluster_card(cluster_cards, SELECTED_TOPIC_ID)" in source
    assert "original_mean_hdbscan_probability" in source
    assert "mean_reassignment_cosine_similarity" in source
    assert "reassignment_exclusion_reason" in source
    assert "TOP_EXAMPLES_PER_CLUSTER = None" in source
    assert "def show_representative_comments(" in source
    assert "representative_comments(artifacts, n=n, topic_id=topic_id)" in source
    assert "if not SHOW_PRIVATE_TEXT" in source
    assert "centroid_similarity" in source
    assert "hdbscan_probability" in source
    assert "REVIEW_EXAMPLES_PER_CLUSTER = 10" in source
    assert "def show_assignment_review_comments(" in source
    assert "assignment_review_comments(artifacts, n=n, topic_id=topic_id)" in source
    assert "low_hdbscan_probability" not in source
    assert "best_cosine_similarity" in source
    assert "similarity_margin" in source
    assert "candidate topic ID" in source
    assert "EXPLAIN_RECORD_INDEX = None" in source
    assert "def show_assignment_explanation(" in source
    assert "result = explain_assignment(" in source
    assert "Representative topic context (not the mathematical cause of assignment)" in source
    assert 'hyperlinks="html"' in source
    assert "Topic keyword similarity" in source
    assert "top_video_share" in source
    assert "Language composition by topic" in source
    assert "Largest topic activity by month" in source
    assert "manual-topic-review.csv" in source
    assert "report-manifest.json" in source
