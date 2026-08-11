from scripts.mine_topics import clean, clustering_prefix


def test_model_specific_clustering_prefixes() -> None:
    assert clustering_prefix("intfloat/multilingual-e5-large") == "query: "
    assert clustering_prefix("ai-forever/FRIDA") == "categorize_topic: "
    assert clustering_prefix("deepvk/USER-bge-m3") == ""


def test_clean_with_spam_filter_uses_pipeline_noise_gate() -> None:
    rows = [
        {"text": "Нам нужен простой CRM для работы с клиентами"},
        {"text": "оченьдлинноесклеенноесловобезпробелов"},
    ]

    assert clean(rows, min_length=10, spam_filter=True) == [rows[0]]
