from pathlib import Path

from fastapi.templating import Jinja2Templates

from src.web.formatting import format_number, humanize_ru

TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "web" / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.filters["humanize_ru"] = humanize_ru
templates.env.filters["format_number"] = format_number
templates.env.globals["min"] = min
