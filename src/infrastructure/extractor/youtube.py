import logging
from datetime import datetime
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from src.domain.document import Document
from src.domain.source import Source, SourceType
from src.infrastructure.api.youtube import YoutubeClient
from src.infrastructure.extractor.base import Extractor

logger = logging.getLogger(__name__)


class YoutubeExtractor(Extractor):
    """Extracts YouTube videos matching a search query and maps them to Source objects."""

    def __init__(self, client: YoutubeClient) -> None:
        self.client = client

    def extract_sources(self, query: str, limit: int = 50) -> list[Source]:
        """Search for videos and fetch their details, mapping results to Source objects.

        Args:
            query (str): The search query.
            limit (int, optional): The maximum number of videos to extract. Defaults to 50.

        Returns:
            list[Source]: Extracted sources.

        """
        search_results = self.client.search_videos(query, limit)
        video_ids = [cast("str", cast("dict[str, Any]", item["id"])["videoId"]) for item in search_results]

        video_details = self.client.get_video_details(video_ids)
        sources = [self._to_source(video) for video in video_details]

        logger.info("Extracted %d source(s) for query=%r", len(sources), query)
        return sources

    def extract_documents(self, source: Source, limit: int = 100) -> list[Document]:
        """Fetch top-level comments for a video source and map them to Document objects.

        Args:
            source (Source): A persisted YouTube video source (must have an id).
            limit (int, optional): The maximum number of documents to extract. Defaults to 100.

        Returns:
            list[Document]: Extracted documents.

        """
        if source.id is None:
            msg = "Source must be persisted (have an id) before extracting documents"
            raise ValueError(msg)

        video_id = self._parse_video_id(source.url)
        comment_threads = self.client.get_comment_threads(video_id, limit)
        documents = [self._to_document(source.id, thread) for thread in comment_threads]

        logger.info("Extracted %d document(s) for source_id=%d", len(documents), source.id)
        return documents

    @staticmethod
    def _parse_video_id(url: str) -> str:
        return parse_qs(urlparse(url).query)["v"][0]

    @staticmethod
    def _to_document(source_id: int, thread: dict[str, Any]) -> Document:
        thread_snippet = cast("dict[str, Any]", thread["snippet"])
        top_level_comment = cast("dict[str, Any]", thread_snippet["topLevelComment"])
        comment_snippet = cast("dict[str, Any]", top_level_comment["snippet"])

        return Document(
            source_id=source_id,
            external_id=top_level_comment["id"],
            text=comment_snippet.get("textDisplay", ""),
            created_at=datetime.fromisoformat(comment_snippet["publishedAt"]),
            metadata={
                "author_display_name": comment_snippet.get("authorDisplayName"),
                "like_count": comment_snippet.get("likeCount", 0),
                "total_reply_count": thread_snippet.get("totalReplyCount", 0),
            },
        )

    @staticmethod
    def _parse_statistics(stats: dict[str, Any]) -> dict[str, int]:
        return {
            "view_count": int(stats.get("viewCount", 0)),
            "like_count": int(stats.get("likeCount", 0)),
            "comment_count": int(stats.get("commentCount", 0)),
            "favorite_count": int(stats.get("favoriteCount", 0)),
        }

    @staticmethod
    def _parse_content_details(details: dict[str, Any]) -> dict[str, Any]:
        return {
            "duration": details.get("duration"),
            "definition": details.get("definition"),
            "has_captions": details.get("caption") == "true",
            "licensed_content": details.get("licensedContent", False),
        }

    @staticmethod
    def _to_source(video: dict[str, Any]) -> Source:
        snippet = cast("dict[str, Any]", video.get("snippet", {}))
        video_id = video["id"]

        return Source(
            type=SourceType.YOUTUBE_VIDEO,
            url=f"https://www.youtube.com/watch?v={video_id}",
            name=snippet.get("title", ""),
            metadata={
                "description": snippet.get("description", ""),
                "published_at": snippet.get("publishedAt"),
                "channel_id": snippet.get("channelId"),
                "channel_title": snippet.get("channelTitle"),
                "statistics": YoutubeExtractor._parse_statistics(
                    cast("dict[str, Any]", video.get("statistics", {})),
                ),
                "content_details": YoutubeExtractor._parse_content_details(
                    cast("dict[str, Any]", video.get("contentDetails", {})),
                ),
            },
        )
