from src.analysis import CleaningReason, CommentRecord, classify, clean


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
