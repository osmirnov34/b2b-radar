from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


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
