import logging
from http import HTTPStatus
from typing import Any, cast

from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


class YoutubeClient:
    """Client for YouTube platform using official Data API v3."""

    def __init__(self, api_keys: list[str]) -> None:
        self.api_keys = api_keys
        self.current_key_index = 0
        self.youtube = self._build_client()
        logger.info("YoutubeClient initialized with %d API key(s)", len(api_keys))

    def _build_client(self) -> Resource:
        logger.debug("Building YouTube client with key index %s", self.current_key_index)
        return build("youtube", "v3", developerKey=self.api_keys[self.current_key_index])

    @staticmethod
    def _is_quota_error(error: HttpError) -> bool:
        """Check if the error is due to quota exhaustion."""
        if error.status_code != HTTPStatus.FORBIDDEN:
            return False
        details = error.error_details
        if not isinstance(details, list):
            return False
        return any(isinstance(detail, dict) and detail.get("reason") == "quotaExceeded" for detail in details)

    def next_key(self) -> bool:
        """Switch to the next API key.

        Returns:
            bool: True if switched to the next key, False if no more keys available.

        """
        if self.current_key_index >= len(self.api_keys) - 1:
            logger.warning("No more API keys available to switch to")
            return False
        self.current_key_index += 1
        self.youtube = self._build_client()
        logger.info("Quota exceeded, switched to key index %s", self.current_key_index)
        return True

    def search_videos(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search for videos on YouTube with pagination support.

        Args:
            query (str): The search query.
            limit (int, optional): The maximum number of results to return. Defaults to 50.

        Returns:
            list[dict[str, Any]]: A list of video search results.

        Raises:
            HttpError: If quota is exceeded and no more API keys available.

        """
        logger.info("Searching videos for query=%r, limit=%d", query, limit)
        results: list[dict[str, Any]] = []
        next_page_token: str | None = None

        while len(results) < limit:
            batch_size = min(50, limit - len(results))

            try:
                request = self.youtube.search().list(
                    q=query,
                    part="snippet",
                    maxResults=batch_size,
                    type="video",
                    pageToken=next_page_token,
                    order="relevance",
                )
                response = cast("dict[str, Any]", request.execute())
            except HttpError as e:
                if self._is_quota_error(e) and self.next_key():
                    continue
                logger.exception("YouTube API request failed for query=%r", query)
                raise

            items = cast("list[dict[str, Any]]", response.get("items", []))
            if not items:
                break

            results.extend(items)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        logger.info("Search for query=%r returned %d result(s)", query, len(results))
        return results

    def get_video_details(self, video_ids: list[str]) -> list[dict[str, Any]]:
        """Get detailed information for a list of videos.

        Args:
            video_ids (list[str]): List of YouTube video IDs.

        Returns:
            list[dict[str, Any]]: Detailed video resources including snippet, statistics, and contentDetails.

        """
        if not video_ids:
            return []

        logger.info("Fetching details for %d video(s)", len(video_ids))
        video_details: list[dict[str, Any]] = []

        # YouTube API limit: up to 50 video IDs per request
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i : i + 50]

            while True:
                try:
                    request = self.youtube.videos().list(
                        id=",".join(batch_ids),
                        part="snippet,statistics,contentDetails",
                    )
                    response = cast("dict[str, Any]", request.execute())
                    items = cast("list[dict[str, Any]]", response.get("items", []))
                    video_details.extend(items)
                    break
                except HttpError as e:
                    if self._is_quota_error(e) and self.next_key():
                        logger.warning("Quota hit during details fetch, switched key and retrying batch")
                        continue
                    logger.exception("Failed to fetch video details")
                    raise

        return video_details
