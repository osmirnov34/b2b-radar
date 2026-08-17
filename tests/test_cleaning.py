from src.ml import CleaningConfig, CleaningReason, CommentRecord, classify, clean
from src.ml.cleaning import classify_prepared_text, prepare_text


def test_classify_flags_empty_too_short_acknowledgement_and_foreign_language() -> None:
    assert classify("   ", min_length=1).reason == CleaningReason.EMPTY
    assert classify("ok", min_length=10).reason == CleaningReason.TOO_SHORT
    assert classify("спасибо класс супер", min_length=1).reason == CleaningReason.ACKNOWLEDGEMENT
    assert classify("This is an English comment", min_length=1).reason == CleaningReason.FOREIGN_LANGUAGE


def test_classify_keeps_meaningful_russian_text() -> None:
    decision = classify("Нам нужен простой CRM для работы с клиентами", min_length=10)

    assert decision.keep is True
    assert decision.reason is None


def test_clean_with_spam_filter_uses_pipeline_noise_gate() -> None:
    rows = [
        CommentRecord(text="Нам нужен простой CRM для работы с клиентами"),
        CommentRecord(text="оченьдлинноесклеенноесловобезпробелов"),
    ]

    result = clean(rows, min_length=10, spam_filter=True)

    assert result.comments == [rows[0]]
    assert result.stats.removed_by_reason[CleaningReason.PIPELINE_NOISE] == 1


def test_clean_collapses_case_and_whitespace_exact_duplicates() -> None:
    rows = [
        CommentRecord(text="CRM  для бизнеса"),
        CommentRecord(text=" crm для БИЗНЕСА "),
    ]

    result = clean(rows, min_length=1)

    assert result.comments == [rows[0]]
    assert result.stats.removed_by_reason[CleaningReason.EXACT_DUPLICATE] == 1


def test_clean_stats_add_up_to_input() -> None:
    rows = [
        CommentRecord(text="Нам нужен простой CRM для работы с клиентами"),
        CommentRecord(text=""),
        CommentRecord(text="спасибо"),
    ]

    result = clean(rows, min_length=1)

    assert result.stats.n_input == 3
    assert result.stats.n_kept == 1
    assert sum(result.stats.removed_by_reason.values()) == 2


def test_prepare_text_normalizes_unicode_html_spaces_and_urls() -> None:
    config = CleaningConfig(min_length=1, url_handling="token")

    prepared = prepare_text("  CRM\u00a0&lt;b&gt;не работает&lt;/b&gt; https://example.com/x  ", config)

    assert prepared.clean_text == "CRM не работает <URL>"
    assert prepared.normalized_text_key == "crm не работает url"


def test_cleaning_keeps_meaningful_thanks_and_rejects_url_only() -> None:
    config = CleaningConfig(min_length=1)
    meaningful = prepare_text("Спасибо, но интеграция с CRM не работает", config)
    url_only = prepare_text("https://example.com/path", config)

    assert classify_prepared_text(meaningful.clean_text, meaningful, "russian", config).keep
    assert (
        classify_prepared_text("https://example.com/path", url_only, "unknown", config).reason
        == CleaningReason.URL_ONLY
    )


def test_cleaning_filters_configured_language_repetition_and_uppercase() -> None:
    language_config = CleaningConfig(min_length=1, allowed_languages=("russian",))
    repeated_config = CleaningConfig(min_length=1, repeated_char_filter=True)
    uppercase_config = CleaningConfig(min_length=1, repeated_char_filter=False, uppercase_filter=True)
    normal = prepare_text("This is a detailed English comment", language_config)
    repeated = prepare_text("Не работает!!!!!!", repeated_config)
    uppercase = prepare_text("ВСЕ ОЧЕНЬ ДОРОГО", uppercase_config)

    assert classify_prepared_text(normal.clean_text, normal, "english", language_config).reason == (
        CleaningReason.DISALLOWED_LANGUAGE
    )
    assert classify_prepared_text(repeated.clean_text, repeated, "russian", repeated_config).reason == (
        CleaningReason.REPEATED_CHARACTERS
    )
    assert classify_prepared_text(uppercase.clean_text, uppercase, "russian", uppercase_config).reason == (
        CleaningReason.MOSTLY_UPPERCASE
    )
