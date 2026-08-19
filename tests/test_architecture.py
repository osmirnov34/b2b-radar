import ast
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
ML_STAGE_SCRIPTS = {
    "build_corpus.py",
    "build_topic_representations.py",
    "clean_dataset.py",
    "cluster_corpus.py",
    "evaluate_topics.py",
    "export_topic_results.py",
    "generate_embeddings.py",
    "inspect_dataset.py",
    "reassign_outliers.py",
    "reduce_dimensions.py",
    "run_eda.py",
    "semantic_deduplicate.py",
    "split_dataset.py",
}
HEAVY_ML_PACKAGES = {"bertopic", "hdbscan", "hnswlib", "numpy", "scipy", "sklearn", "torch", "umap"}


def _python_sources(relative: str) -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in (PROJECT_ROOT / relative).rglob("*.py"))


def test_ml_does_not_depend_on_ingestion_web_or_infrastructure() -> None:
    sources = _python_sources("src/ml")

    assert "src.ingestion" not in sources
    assert "src.web" not in sources
    assert "src.infrastructure" not in sources


def test_ingestion_does_not_depend_on_ml() -> None:
    assert "src.ml" not in _python_sources("src/ingestion")


def test_shared_text_processing_has_no_outer_layer_dependencies() -> None:
    sources = _python_sources("src/text_processing")

    assert "src.ml" not in sources
    assert "src.ingestion" not in sources
    assert "src.web" not in sources
    assert "src.infrastructure" not in sources


def test_legacy_monolithic_ml_entrypoint_is_absent() -> None:
    assert not (PROJECT_ROOT / "scripts/mine_topics.py").exists()
    assert not (PROJECT_ROOT / "configs/topic-analysis.example.json").exists()


def test_scripts_do_not_import_removed_legacy_models() -> None:
    sources = _python_sources("scripts")

    assert "AnalysisConfig" not in sources
    assert "AnalysisResult" not in sources


def test_ml_stage_scripts_are_thin_adapters_without_heavy_ml_imports() -> None:
    for name in ML_STAGE_SCRIPTS:
        tree = ast.parse((PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8"))
        definitions = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        imported_roots = {
            alias.name.split(".", maxsplit=1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_roots.update(
            node.module.split(".", maxsplit=1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        )

        assert definitions <= {"build_parser", "main"}, name
        assert imported_roots.isdisjoint(HEAVY_ML_PACKAGES), name


def test_development_notebook_keeps_expensive_execution_explicit() -> None:
    notebook = json.loads((PROJECT_ROOT / "notebooks/01_development_eda.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "RUN_SMOKE = False" in source
    assert "run_smoke_pipeline(" in source
    assert re.search(r"\brun_pipeline\(", source) is None
    assert "subprocess" not in source
