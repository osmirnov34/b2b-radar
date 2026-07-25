from src.domain.document import Document
from src.domain.filter_settings import FilterSettings


class YoutubeQualityFilter:
    """Applies :class:`FilterSettings` thresholds to YouTube comment metadata."""

    def __init__(self, settings: FilterSettings) -> None:
        self.settings = settings

    def accepts_comment(self, document: Document) -> bool:
        """Return True if the comment clears every enabled comment quality threshold."""
        if len(document.text.strip()) < self.settings.document_min_length:
            return False
        if document.metadata.get("like_count", 0) < self.settings.document_min_likes:
            return False
        return int(document.metadata.get("total_reply_count", 0)) >= self.settings.document_min_replies
