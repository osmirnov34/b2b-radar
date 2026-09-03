import ast
import json
import re
from pathlib import Path

from src.operations import PipelineStatus, run_pipeline

PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/00_colab_pipeline.ipynb"


def _notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_sources(notebook: dict[str, object]) -> list[str]:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    return ["".join(cell.get("source", [])) for cell in cells if cell.get("cell_type") == "code"]


def test_colab_notebook_is_clean_and_all_code_cells_parse() -> None:
    notebook = _notebook()
    sources = _code_sources(notebook)

    assert notebook["nbformat"] == 4
    assert sources
    for source in sources:
        ast.parse(source)
    for cell in notebook["cells"]:
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []


def test_colab_notebook_keeps_expensive_and_destructive_actions_guarded() -> None:
    source = "\n".join(_code_sources(_notebook()))

    assert "RUN_SMOKE = False" in source
    assert "RUN_FULL = False" in source
    assert "if RUN_SMOKE:" in source
    assert "if RUN_FULL:" in source
    assert "dry_report.can_run" in source
    assert source.count("echo_progress=True") == 2
    assert "smoke_report.full_run_allowed" in source
    assert "publish_snapshot" not in source
    assert "rm -rf" not in source
    assert "reset --hard" not in source


def test_colab_notebook_pins_code_model_and_uses_public_operations_api() -> None:
    source = "\n".join(_code_sources(_notebook()))

    assert "https://github.com/osmirnov34/b2b-radar.git" in source
    assert "1RreCEUjXYy1qSB0N66o0W6XuWVsD5xYN" in source
    assert re.search(r'MODEL_REVISION = "[0-9a-f]{40}"', source)
    assert 'CODE_REF = "main"' in source
    assert '"-e", f"{project_root}[analysis]"' in source
    assert "from src.operations import (" in source
    assert "from scripts" not in source
    assert run_pipeline is not None
    assert PipelineStatus.COMPLETED.value == "completed"


def test_colab_notebook_supports_python_313_and_checks_binary_ml_stack() -> None:
    source = "\n".join(_code_sources(_notebook()))

    assert "(3, 13)" in source
    assert "expected 3.11, 3.12, or 3.13" in source
    assert '"pip", "install", "--upgrade", "pip", "setuptools", "wheel"' in source
    assert 'shutil.which("g++")' in source
    assert '"hnswlib==0.8.0", "--no-binary=hnswlib"' in source
    assert '"hnswlib_install": hnswlib_install' in source
    for package in ("sentence_transformers", "umap", "hdbscan", "hnswlib"):
        assert f'importlib.import_module("{package}")' in source
    assert "compatibility_check" in source
    assert source.index("inspect_comments_jsonl(") < source.index("compatibility_vectors")
    assert "colab-environment.json" in source


def test_colab_notebook_validates_before_creating_runtime_config() -> None:
    sources = _code_sources(_notebook())
    combined = "\n".join(sources)

    assert combined.index("inspect_comments_jsonl(") < combined.index("PipelineConfig.model_validate_json(")
    assert "DatasetFormat.JSONL" in combined
    assert "inspection.is_usable" in combined
    assert "RECORD_COUNT_TOLERANCE = 0.25" in combined
    assert "expected_records_tolerance=RECORD_COUNT_TOLERANCE" in combined
    assert '"record_count_comparison"' in combined
    assert '"expected_records": EXPECTED_RECORDS' in combined
    assert '"expected_records_tolerance": RECORD_COUNT_TOLERANCE' in combined
    assert '"require_final_evaluation": true' not in combined
    assert "embeddings.colab.json" in combined
    assert '"device": "cuda"' in combined
