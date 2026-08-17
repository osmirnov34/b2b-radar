from src.text_processing.language import detect_language, is_probably_russian
from src.text_processing.noise import is_noise


def test_short_text_is_kept_when_language_is_uncertain() -> None:
    assert is_probably_russian("Короткое название")


def test_language_detection_returns_stable_labels() -> None:
    assert detect_language("Нам нужна простая система для работы с клиентами") == "russian"
    assert detect_language("short") == "unknown_short"


def test_russian_text_is_kept() -> None:
    assert is_probably_russian("Это подробное описание сервиса для российского малого бизнеса")


def test_english_text_is_rejected() -> None:
    assert not is_probably_russian("This is a detailed description of software for small business owners")


def test_noise_filter_rejects_empty_emoji_wall_and_gibberish() -> None:
    assert is_noise("   ")
    assert is_noise("🚀🚀🚀🚀🚀 да")
    assert is_noise("оченьдлинноесклеенноесловобезпробелов")


def test_noise_filter_keeps_normal_comment() -> None:
    assert not is_noise("Нам нужен простой CRM для работы с клиентами.")
