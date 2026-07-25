import re

# Gopher symbol/word ratios

_WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)  # noqa: RUF001 — Cyrillic range is intentional
_PUNCT = frozenset(".,!?;:—–-«»\"“”'’‘`()[]{}…%№/\\+*=&@#_|^~<>$ \t\n\r")  # noqa: RUF001 — typographic punctuation is intentional

_MAX_SYMBOL_RATIO = 0.20  # share of emoji/glyph chars — emoji-wall spam
_MAX_MEAN_WORD_LEN = 14  # avg word length; huge => glued gibberish
_MAX_WORD_LEN = 25  # a single monster token the mean would hide


def _is_symbol(ch: str) -> bool:
    return not (ch.isalnum() or ch in _PUNCT)


def is_noise(text: str) -> bool:
    """Return True for junk a language gate lets through: emoji walls and glued gibberish."""
    stripped = text.strip()
    if not stripped:
        return True

    n_symbol = sum(1 for ch in stripped if _is_symbol(ch))
    if n_symbol / len(stripped) > _MAX_SYMBOL_RATIO:
        return True

    words = _WORD_RE.findall(stripped.lower())
    if words:
        mean_len = sum(len(word) for word in words) / len(words)
        if mean_len > _MAX_MEAN_WORD_LEN or max(len(word) for word in words) > _MAX_WORD_LEN:
            return True

    return False
