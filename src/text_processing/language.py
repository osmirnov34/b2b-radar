from functools import lru_cache

from lingua import Language, LanguageDetector, LanguageDetectorBuilder

# lingua only returns a language from this set, so the Cyrillic neighbours must be here to be
# recognised (and dropped) instead of collapsing onto Russian, and the Latin languages give
# foreign titles somewhere to land. Other scripts (Han, Arabic, ...) yield no match -> not Russian.
_DETECTOR_LANGS = (
    Language.RUSSIAN,
    Language.UKRAINIAN,
    Language.BULGARIAN,
    Language.BELARUSIAN,
    Language.SERBIAN,
    Language.KAZAKH,
    Language.ENGLISH,
    Language.GERMAN,
    Language.FRENCH,
    Language.SPANISH,
    Language.TURKISH,
    Language.POLISH,
)

# Below this many characters there is too little signal to trust a verdict, so we keep the source
# rather than risk dropping a real Russian video on a noisy few-word (often brand-only) title.
_MIN_CHARS_FOR_DETECTION = 20


@lru_cache(maxsize=1)
def _detector() -> LanguageDetector:
    """Build the shared detector once; lingua loads its FST models lazily on first use."""
    return LanguageDetectorBuilder.from_languages(*_DETECTOR_LANGS).build()


def is_probably_russian(*parts: str | None, min_chars: int = _MIN_CHARS_FOR_DETECTION) -> bool:
    """Keep a source only when the combined text (e.g. video title + description) is Russian.

    Ukrainian, other Cyrillic languages and foreign-script text are dropped. Text shorter than
    ``min_chars`` is kept: too little signal to risk dropping a real Russian source.
    """
    text = " ".join(part.strip() for part in parts if part).strip()
    if len(text) < min_chars:
        return True
    return _detector().detect_language_of(text) == Language.RUSSIAN


def detect_language(text: str, *, min_chars: int = _MIN_CHARS_FOR_DETECTION) -> str:
    """Return a stable language label without using the result as a filtering decision."""
    stripped = text.strip()
    if len(stripped) < min_chars:
        return "unknown_short"
    language = _detector().detect_language_of(stripped)
    return language.name.lower() if language is not None else "unknown"
