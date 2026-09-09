import ast
import json
from pathlib import Path
from typing import Any, cast

NOTEBOOK_PATH = Path(__file__).parents[1] / "notebooks/02_colab_results_analysis.ipynb"


def _notebook() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8")))


def test_analysis_notebook_has_guarded_interactive_report() -> None:
    notebook = _notebook()
    cells = notebook["cells"]
    code = ["".join(cell.get("source", [])) for cell in cells if cell.get("cell_type") == "code"]
    source = "\n".join(code)

    for cell_source in code:
        ast.parse(cell_source)
    assert "WRITE_INTERACTIVE_REPORT = False" in source
    assert "INTERACTIVE_REPORT_PRIVATE_EXAMPLES = False" in source
    assert "write_interactive_report(" in source
    assert 'Path(DRIVE_PROJECT_DIR) / "interactive-reports" / RUN_ID' in source
    assert "include_private_examples=INTERACTIVE_REPORT_PRIVATE_EXAMPLES" in source
    assert "private_text_included" in source


def test_analysis_notebook_remains_clean_and_does_not_mutate_pipeline() -> None:
    notebook = _notebook()
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")

    for cell in notebook["cells"]:
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
    assert "run_pipeline(" not in source
    assert "run_smoke_pipeline(" not in source
