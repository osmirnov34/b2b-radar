import logging
import socket
import ssl
import time
from http import HTTPStatus
from http.client import HTTPException
from typing import Any, cast

from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError
from googleapiclient.http import HttpRequest

logger = logging.getLogger(__name__)

# Transient HTTP statuses that are safe to retry (server-side / rate-limit hiccups).
_RETRYABLE_STATUSES = frozenset({500, 502, 503, 504})
# Network-level failures raised before any HTTP response is available.
_RETRYABLE_NETWORK_ERRORS = (socket.timeout, ConnectionError, TimeoutError, ssl.SSLError, HTTPException)
_MAX_ATTEMPTS = 4
_INITIAL_BACKOFF_SECONDS = 1.0


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
    def _execute_with_retry(request: HttpRequest) -> dict[str, Any]:
        """Execute a YouTube API request, retrying transient 5xx and network errors with backoff.

        Quota (``quotaExceeded``) and ``commentsDisabled`` are 403s and are never retried here;
        they are re-raised for the caller to handle (key rotation / graceful skip).
        """
        backoff = _INITIAL_BACKOFF_SECONDS
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return cast("dict[str, Any]", request.execute())
            except HttpError as error:
                if error.status_code not in _RETRYABLE_STATUSES or attempt == _MAX_ATTEMPTS:
                    raise
                logger.warning(
                    "Transient YouTube API error (status=%s), attempt %d/%d, retrying in %.1fs",
                    error.status_code,
                    attempt,
                    _MAX_ATTEMPTS,
                    backoff,
                )
            except _RETRYABLE_NETWORK_ERRORS as error:
                if attempt == _MAX_ATTEMPTS:
                    raise
                logger.warning(
                    "Transient network error (%s), attempt %d/%d, retrying in %.1fs",
                    type(error).__name__,
                    attempt,
                    _MAX_ATTEMPTS,
                    backoff,
                )
            time.sleep(backoff)
            backoff *= 2
        msg = "Retry loop exited without returning a response"  # pragma: no cover - defensive
        raise RuntimeError(msg)  # pragma: no cover

    @staticmethod
    def _has_error_reason(error: HttpError, reason: str) -> bool:
        if error.status_code != HTTPStatus.FORBIDDEN:
            return False
        details = error.error_details
        if not isinstance(details, list):
            return False
        return any(isinstance(detail, dict) and detail.get("reason") == reason for detail in details)

    @classmethod
    def _is_quota_error(cls, error: HttpError) -> bool:
        """Check if the error is due to quota exhaustion."""
        return cls._has_error_reason(error, "quotaExceeded")

    @classmethod
    def _is_comments_disabled_error(cls, error: HttpError) -> bool:
        """Check if the error is due to comments being disabled on the video."""
        return cls._has_error_reason(error, "commentsDisabled")

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
                response = self._execute_with_retry(request)
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
                    response = self._execute_with_retry(request)
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

    def get_comment_threads(self, video_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch top-level comment threads for a video.

        Args:
            video_id (str): The YouTube video ID.
            limit (int, optional): The maximum number of comment threads to return. Defaults to 100.

        Returns:
            list[dict[str, Any]]: Comment thread resources, or an empty list if comments are disabled.

        Raises:
            HttpError: If quota is exceeded and no more API keys available.

        """
        logger.info("Fetching comment threads for video_id=%s, limit=%d", video_id, limit)
        results: list[dict[str, Any]] = []
        next_page_token: str | None = None

        while len(results) < limit:
            batch_size = min(100, limit - len(results))

            try:
                request = self.youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=video_id,
                    maxResults=batch_size,
                    pageToken=next_page_token,
                    textFormat="plainText",
                )
                response = self._execute_with_retry(request)
            except HttpError as e:
                if self._is_quota_error(e) and self.next_key():
                    continue
                if self._is_comments_disabled_error(e):
                    logger.info("Comments disabled for video_id=%s", video_id)
                    return results
                logger.exception("Failed to fetch comment threads for video_id=%s", video_id)
                raise

            items = cast("list[dict[str, Any]]", response.get("items", []))
            if not items:
                break

            results.extend(items)

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        logger.info("Fetched %d comment thread(s) for video_id=%s", len(results), video_id)
        return results
