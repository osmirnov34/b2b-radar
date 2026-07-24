import re
from datetime import UTC, datetime
from typing import Any

from src.domain.document import Document
from src.domain.filter_settings import FilterSettings
from src.domain.source import Source

_DURATION_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def _parse_iso8601_duration(value: str | None) -> int | None:
    """Convert a YouTube ISO 8601 duration (e.g. 'PT5M30S') into whole seconds."""
    if not value:
        return None
    match = _DURATION_RE.fullmatch(value)
    if match is None:
        return None
    hours, minutes, seconds = (int(part) if part else 0 for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


class YoutubeQualityFilter:
    """Applies: class:`FilterSettings` thresholds to YouTube video/comment metadata."""

    def __init__(self, settings: FilterSettings) -> None:
        self.settings = settings

    def accepts_video(self, source: Source) -> bool:  # noqa: PLR0911
        """Return True if the video clears every enabled video quality threshold."""
        statistics = self._as_dict(source.metadata.get("statistics"))

        if statistics.get("view_count", 0) < self.settings.source_min_views:
            return False
        if statistics.get("like_count", 0) < self.settings.source_min_likes:
            return False
        if statistics.get("comment_count", 0) < self.settings.source_min_comments:
            return False

        if self.settings.source_min_duration_seconds > 0:
            content_details = self._as_dict(source.metadata.get("content_details"))
            duration = _parse_iso8601_duration(content_details.get("duration"))
            if duration is None or duration < self.settings.source_min_duration_seconds:
                return False

        if self.settings.source_max_age_days > 0:
            published_at = _parse_datetime(source.metadata.get("published_at"))
            if published_at is None:
                return False
            age_days = (datetime.now(UTC) - published_at).days
            if age_days > self.settings.source_max_age_days:
                return False

        return True

    def accepts_comment(self, document: Document) -> bool:
        """Return True if the comment clears every enabled comment quality threshold."""
        if len(document.text.strip()) < self.settings.document_min_length:
            return False
        if document.metadata.get("like_count", 0) < self.settings.document_min_likes:
            return False
        return int(document.metadata.get("total_reply_count", 0)) >= self.settings.document_min_replies

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:  # noqa: ANN401
        return value if isinstance(value, dict) else {}
