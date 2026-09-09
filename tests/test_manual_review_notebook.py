import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/04_colab_manual_evaluation.ipynb"


def _sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]


def test_manual_review_notebook_is_clean_and_parses() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    for source in _sources():
        ast.parse(source)
    for cell in notebook["cells"]:
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []


def test_manual_review_notebook_is_private_and_non_mutating_by_default() -> None:
    source = "\n".join(_sources())

    assert "SHOW_PRIVATE_TEXT = False" in source
    assert "SAVE_ANNOTATIONS = False" in source
    assert "OVERWRITE_ANNOTATIONS = False" in source
    assert 'Path(DRIVE_PROJECT_DIR) / "manual-reviews" / RUN_ID / REVIEWER' in source
    assert "run_pipeline(" not in source
    assert "run_smoke_pipeline(" not in source
    assert "force=True" not in source
    assert "reset --hard" not in source
    assert "rm -rf" not in source


def test_manual_review_notebook_supports_resume_validation_and_passport() -> None:
    source = "\n".join(_sources())

    assert "load_manual_review_bundle(run_dir, annotations_path if annotations_path.is_file() else None)" in source
    assert "def show_review_item(" in source
    assert '"""Display one verified sample row and its assignment context after privacy opt-in."""' in source
    assert "def set_annotation(" in source
    assert '"""Validate and stage one pipeline-compatible judgment in notebook memory."""' in source
    assert "validate_manual_annotations(bundle.sample" in source
    assert "reviewer_agreement(reviewer_sets)" in source
    assert "save_manual_annotations(" in source
    assert "build_experiment_passport(" in source
    assert "write_experiment_passport(" in source
    assert "restricted_manual_review" in source
