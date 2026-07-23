from datetime import UTC, datetime

_MINUTE = 60
_HOUR = 3600
_DAY = 86400


def humanize_ru(value: datetime | None) -> str:  # noqa: PLR0911
    """Render a timestamp as a short relative Russian time label, e.g. '2 часа назад'."""
    if value is None:
        return "—"

    now = datetime.now(UTC)
    reference = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    seconds = (now - reference).total_seconds()

    if seconds < _MINUTE:
        return "только что"
    if seconds < _HOUR:
        minutes = int(seconds // _MINUTE)
        return f"{minutes} мин назад"
    if seconds < _DAY:
        hours = int(seconds // _HOUR)
        return f"{hours} ч назад"
    days = int(seconds // _DAY)
    if days == 1:
        return "вчера"
    if days < 7:  # noqa: PLR2004
        return f"{days} дн назад"
    return reference.strftime("%d.%m.%Y")


def format_number(value: int) -> str:
    return f"{value:,}"
