import py3langid as langid

# py3langid confuses the Cyrillic-script languages with each other (a Russian title routinely
# classifies as ``bg``/``uk``), so the gate keeps the whole Cyrillic family rather than only ``ru``.
# The goal here is coarse: drop clearly foreign videos (English music/reviews, emoji spam) before
# spending comment-fetch quota on them, NOT to tell Russian from Ukrainian.
_CYRILLIC_LANGS = frozenset({"ru", "uk", "bg", "mk", "sr", "be", "kk", "ky", "mn"})

# Below this many characters there is too little signal to trust a verdict, so we keep the source
# rather than risk dropping a real Russian video on a noisy few-word (often brand-only) title.
_MIN_CHARS_FOR_DETECTION = 20


def is_probably_russian(*parts: str | None, min_chars: int = _MIN_CHARS_FOR_DETECTION) -> bool:
    """Best-effort language gate over the concatenated text parts (e.g. video title + description).

    Conservative by design: returns ``True`` (keep) unless the combined text is long enough to
    judge AND classifies as a non-Cyrillic language. Callers pass title and description together so
    a Cyrillic description can rescue a brand-heavy Latin title.
    """
    text = " ".join(part.strip() for part in parts if part).strip()
    if len(text) < min_chars:
        return True
    lang, _score = langid.classify(text)
    return lang in _CYRILLIC_LANGS
