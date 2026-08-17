"""Noise filtering and exact-duplicate removal for raw comments.

Near-duplicate removal (on embeddings) is a separate, later stage — see
`src/ml/semantic_deduplication.py`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from html import unescape
from typing import TYPE_CHECKING

from src.ml.models import CleaningDecision, CleaningReason, CleaningResult, CleaningStats, CommentRecord
from src.text_processing.noise import is_noise as is_pipeline_noise

if TYPE_CHECKING:
    from src.ml.config import CleaningConfig

# Short acknowledgements that form dense but useless clusters (valid Russian, so gates let them through).
THANKS_TOKENS = {
    "спс",
    "спасибо",
    "благодарю",
    "класс",
    "супер",
    "круто",
    "лайк",
    "огонь",
    "топ",
    "красота",
    "молодец",
    "молодцы",
    "здорово",
    "отлично",
    "браво",
}

WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)  # noqa: RUF001 — Cyrillic range is intentional
CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)  # noqa: RUF001 — Cyrillic range is intentional
LATIN_RE = re.compile(r"[a-z]", re.IGNORECASE)

_MAX_ACKNOWLEDGEMENT_WORDS = 3
_MIN_CYRILLIC_RATIO = 0.5
_HTML_TAG_RE = re.compile(r"<[^>]{0,10000}>")
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<]+", re.IGNORECASE)
_REPEATED_CHAR_RE = re.compile(r"(.)\1{5,}", re.DOTALL)
_MOSTLY_UPPERCASE_RATIO = 0.8


@dataclass(frozen=True, slots=True)
class PreparedText:
    clean_text: str
    normalized_text_key: str
    was_url_only: bool


def _strip_control_characters(text: str) -> str:
    return "".join(" " if unicodedata.category(char).startswith("C") else char for char in text)


def prepare_text(text: str, config: CleaningConfig) -> PreparedText:
    """Prepare model text and a stricter exact-dedup key without changing the source text."""
    prepared = unicodedata.normalize(config.unicode_normalization, unescape(text))
    if config.strip_html:
        prepared = _HTML_TAG_RE.sub(" ", prepared)
    without_urls = _URL_RE.sub(" ", prepared)
    was_url_only = bool(_URL_RE.search(prepared)) and not any(char.isalnum() for char in without_urls)
    if config.url_handling == "remove":
        prepared = without_urls
    elif config.url_handling == "token":
        prepared = _URL_RE.sub(" <URL> ", prepared)
    prepared = " ".join(_strip_control_characters(prepared).split())
    key_chars = (char if char.isalnum() else " " for char in prepared.casefold())
    normalized_key = " ".join("".join(key_chars).split())
    return PreparedText(clean_text=prepared, normalized_text_key=normalized_key, was_url_only=was_url_only)


def classify_prepared_text(
    source_text: str,
    prepared: PreparedText,
    language: str,
    config: CleaningConfig,
) -> CleaningDecision:
    """Apply configured gates in a deterministic order; categories are mutually exclusive."""
    text = prepared.clean_text
    if not text:
        return CleaningDecision(keep=False, reason=CleaningReason.EMPTY)
    if prepared.was_url_only:
        return CleaningDecision(keep=False, reason=CleaningReason.URL_ONLY)
    if len(text) < config.min_length:
        return CleaningDecision(keep=False, reason=CleaningReason.TOO_SHORT)
    if config.max_length is not None and len(text) > config.max_length:
        return CleaningDecision(keep=False, reason=CleaningReason.TOO_LONG)
    if not any(char.isalnum() for char in text):
        return CleaningDecision(keep=False, reason=CleaningReason.NO_ALPHANUMERIC)
    words = WORD_RE.findall(text.casefold())
    if (
        config.acknowledgement_filter
        and words
        and len(words) <= _MAX_ACKNOWLEDGEMENT_WORDS
        and all(word in THANKS_TOKENS for word in words)
    ):
        return CleaningDecision(keep=False, reason=CleaningReason.ACKNOWLEDGEMENT)
    if config.repeated_char_filter and _REPEATED_CHAR_RE.search(source_text):
        return CleaningDecision(keep=False, reason=CleaningReason.REPEATED_CHARACTERS)
    letters = [char for char in text if char.isalpha()]
    if config.uppercase_filter and letters and (
        sum(char.isupper() for char in letters) / len(letters) > _MOSTLY_UPPERCASE_RATIO
    ):
        return CleaningDecision(keep=False, reason=CleaningReason.MOSTLY_UPPERCASE)
    if config.spam_filter and is_pipeline_noise(text):
        return CleaningDecision(keep=False, reason=CleaningReason.PIPELINE_NOISE)
    if config.allowed_languages and language not in config.allowed_languages:
        return CleaningDecision(keep=False, reason=CleaningReason.DISALLOWED_LANGUAGE)
    return CleaningDecision(keep=True)


def classify(text: str, min_length: int) -> CleaningDecision:
    """Decide whether a comment is signal or noise, with a reason for removals."""
    stripped = text.strip()
    if not stripped:
        return CleaningDecision(keep=False, reason=CleaningReason.EMPTY)
    if len(stripped) < min_length:
        return CleaningDecision(keep=False, reason=CleaningReason.TOO_SHORT)
    words = WORD_RE.findall(stripped.lower())
    if not words:  # no letters at all (pure emoji/punctuation) — treat like too-short
        return CleaningDecision(keep=False, reason=CleaningReason.TOO_SHORT)
    if len(words) <= _MAX_ACKNOWLEDGEMENT_WORDS and all(w in THANKS_TOKENS for w in words):
        return CleaningDecision(keep=False, reason=CleaningReason.ACKNOWLEDGEMENT)
    n_cyr = len(CYRILLIC_RE.findall(stripped))
    n_lat = len(LATIN_RE.findall(stripped))
    if n_cyr + n_lat > 0 and n_cyr / (n_cyr + n_lat) < _MIN_CYRILLIC_RATIO:
        return CleaningDecision(keep=False, reason=CleaningReason.FOREIGN_LANGUAGE)
    return CleaningDecision(keep=True)


def clean(rows: list[CommentRecord], min_length: int, spam_filter: bool = False) -> CleaningResult:
    """Filter noise and drop exact-duplicate texts (near-dup handled later on embeddings).

    With spam_filter=True, additionally apply the pipeline's cheap noise heuristics —
    drops glued gibberish and emoji/symbol spam before embedding.
    Off by default so the calibrated runs in the methodology (§9) stay reproducible.
    """
    seen: set[str] = set()
    kept: list[CommentRecord] = []
    removed_by_reason: dict[CleaningReason, int] = {}

    for row in rows:
        decision = classify(row.text, min_length)
        if decision.keep and spam_filter and is_pipeline_noise(row.text):
            decision = CleaningDecision(keep=False, reason=CleaningReason.PIPELINE_NOISE)
        if decision.keep and row.normalized_text_key in seen:
            decision = CleaningDecision(keep=False, reason=CleaningReason.EXACT_DUPLICATE)

        if decision.keep:
            seen.add(row.normalized_text_key)
            kept.append(row)
        else:
            assert decision.reason is not None  # noqa: S101 — enforced by CleaningDecision's own validator
            removed_by_reason[decision.reason] = removed_by_reason.get(decision.reason, 0) + 1

    stats = CleaningStats(n_input=len(rows), n_kept=len(kept), removed_by_reason=removed_by_reason)
    return CleaningResult(comments=kept, stats=stats)
