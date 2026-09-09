import ast
import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks/05_colab_topic_search.ipynb"


def _sources() -> list[str]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code"]


def test_topic_search_notebook_is_clean_and_parses() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for source in _sources():
        ast.parse(source)
    for cell in notebook["cells"]:
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []


def test_topic_search_notebook_has_independent_safety_gates() -> None:
    source = "\n".join(_sources())

    assert "RUN_SEARCH = False" in source
    assert "SHOW_RESTRICTED_EVIDENCE = False" in source
    assert "SAVE_RESTRICTED_RESULT = False" in source
    assert "OVERWRITE_RESULT = False" in source
    assert "search_user_topic(" in source
    assert "write_topic_search_result(" in source
    assert 'Path(DRIVE_PROJECT_DIR) / "topic-searches" / RUN_ID / query_id' in source
    assert "run_pipeline(" not in source
    assert "run_smoke_pipeline(" not in source
