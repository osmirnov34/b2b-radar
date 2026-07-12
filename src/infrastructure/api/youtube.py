from http import HTTPStatus
from typing import Any, cast

from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError


class YoutubeClient:
    """Client for YouTube platform using official Data API v3."""

    def __init__(self, api_keys: list[str]) -> None:
        self.api_keys = api_keys
        self.current_key_index = 0
        self.youtube = self._build_client()

    def _build_client(self) -> Resource:
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
            return False
        self.current_key_index += 1
        self.youtube = self._build_client()
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
                raise

            items = cast("list[dict[str, Any]]", response.get("items", []))
            if not items:
                break

            results.extend(items)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return results
