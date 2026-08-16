"""Noise filtering and exact-duplicate removal for raw comments.

Near-duplicate removal (on embeddings) is a separate, later stage — see
src/analysis/dedup.py once it exists.
"""

from __future__ import annotations

import re

from src.analysis.models import CleaningDecision, CleaningReason, CleaningResult, CleaningStats, CommentRecord
from src.infrastructure.extractor.noise import is_noise as is_pipeline_noise

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
