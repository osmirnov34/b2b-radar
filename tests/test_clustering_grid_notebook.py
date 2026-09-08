import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/03_colab_clustering_grid.ipynb"


def _sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]


def test_clustering_grid_notebook_is_clean_and_parses() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    for source in _sources():
        ast.parse(source)
    for cell in notebook["cells"]:
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []


def test_clustering_grid_notebook_is_guarded_resumable_and_source_safe() -> None:
    source = "\n".join(_sources())

    assert "MIN_CLUSTER_SIZES = (50, 100, 150, 250)" in source
    assert "RUN_GRID = False" in source
    assert 'grid_dir = Path(DRIVE_PROJECT_DIR) / "ml-experiments"' in source
    assert "base_config=artifacts.clustering.config" in source
    assert "run_clustering_grid(" in source
    assert "progress=print_grid_progress" in source
    assert "lambda" not in source
    assert "force=True" not in source
    assert "run_pipeline(" not in source
    assert "restart_from" not in source
    assert "MATCH_MINIMUM_OVERLAP_SHARE = 0.05" in source
    assert "match_grid_clusters(" in source
    assert "primary_match" in source
    assert "go.Sankey(" in source
    assert "source_retention" in source
    assert "target_composition" in source
    assert "TOP_STABILITY_TRAJECTORIES = 30" in source
    assert "analyze_grid_stability(" in source
    assert "ClusterStabilityConfig()" in source
    assert "grid_coverage" in source
    assert "minimum_jaccard" in source
    assert "minimum_source_retention" in source
    assert "ambiguous_transition" in source
    assert "Cluster stability levels by grid variant" in source
    assert "Largest mutual-primary cluster trajectories" in source
    assert "HIERARCHY_MINIMUM_PARENT_CONTAINMENT = 0.80" in source
    assert "build_grid_hierarchy(" in source
    assert "primary_edges" in source
    assert "secondary_edges" in source
    assert "nesting_violations" in source
    assert "go.Icicle(" in source
    assert "Empirical primary hierarchy topology (equal node weight)" in source


def test_clustering_grid_notebook_compares_core_metrics() -> None:
    source = "\n".join(_sources())

    for metric in (
        "clusters",
        "outlier_share",
        "mean_probability",
        "low_confidence_share",
        "median_cluster_size",
        "dominant_cluster_share",
        "relative_validity",
        "dbcv",
    ):
        assert f'"{metric}"' in source
