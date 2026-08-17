"""Topic keywords and representatives derived from fixed HDBSCAN labels."""

from __future__ import annotations

import hashlib
import heapq
import json
import pickle
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.ml.clustering import ClusteringManifest
from src.ml.corpus import CorpusManifest, CorpusRecord

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray


class _TopicModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TopicRepresentationConfig(_TopicModel):
    schema_version: int = 1
    min_df: int = Field(default=1, ge=1)
    max_df: float = Field(default=0.95, gt=0, le=1)
    ngram_min: int = Field(default=1, ge=1, le=3)
    ngram_max: int = Field(default=2, ge=1, le=3)
    max_features: int | None = Field(default=50_000, ge=1)
    top_n_words: int = Field(default=10, ge=1, le=100)
    stopword_languages: tuple[str, ...] = ("ru", "en")
    extra_stopwords: tuple[str, ...] = ()
    bm25_weighting: bool = True
    reduce_frequent_words: bool = True
    minimum_token_length: int = Field(default=2, ge=1, le=20)
    representatives_per_topic: int = Field(default=5, ge=1, le=100)
    representative_candidate_multiplier: int = Field(default=20, ge=1, le=1000)
    minimum_representative_probability: float = Field(default=0.5, ge=0, le=1)
    max_cluster_characters: int | None = Field(default=5_000_000, ge=1000)
    similar_topic_jaccard_warning: float = Field(default=0.7, ge=0, le=1)

    @model_validator(mode="after")
    def validate_ngrams(self) -> TopicRepresentationConfig:
        if self.ngram_max < self.ngram_min:
            msg = "ngram_max cannot be less than ngram_min"
            raise ValueError(msg)
        return self

    @field_validator("stopword_languages", "extra_stopwords")
    @classmethod
    def normalize_words(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value.strip().casefold() for value in values if value.strip()))


class TopicKeyword(_TopicModel):
    term: str = Field(min_length=1)
    weight: float = Field(ge=0)
    rank: int = Field(ge=1)
    kind: str


class RepresentativeIndices(_TopicModel):
    topic_id: int = Field(ge=0)
    record_indices: list[int]
    centroid_similarities: list[float]


class TopicRepresentation(_TopicModel):
    topic_id: int = Field(ge=0)
    name: str
    records: int = Field(ge=1)
    mean_probability: float = Field(ge=0, le=1)
    languages: dict[str, int]
    unique_videos: int = Field(ge=0)
    keywords: list[TopicKeyword]
    representative_indices: list[int]


class TopicRepresentationQuality(_TopicModel):
    topics: int = Field(ge=0)
    empty_topics: int = Field(ge=0)
    vocabulary_size: int = Field(ge=0)
    topic_diversity: float = Field(ge=0, le=1)
    maximum_keyword_jaccard: float = Field(ge=0, le=1)
    similar_topic_pairs: int = Field(ge=0)
    duplicate_names: int = Field(ge=0)
    mean_representative_similarity: float | None = Field(default=None, ge=-1, le=1)


class TopicRepresentationManifest(_TopicModel):
    schema_version: int = 1
    corpus_manifest_path: str
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clustering_manifest_path: str
    clustering_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_path: str
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    embeddings_path: str
    embeddings_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    labels_path: str
    labels_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    probabilities_path: str
    probabilities_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_records: int = Field(ge=0)
    source_topics: int = Field(ge=0)
    topics: int = Field(ge=0)
    omitted_topics: int = Field(ge=0)
    outliers: int = Field(ge=0)
    config: TopicRepresentationConfig
    stopwords_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    backend: str
    backend_version: str
    representations_path: str
    representations_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    keywords_path: str
    keywords_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    representative_indices_path: str
    representative_indices_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocabulary_path: str
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ctfidf_path: str
    ctfidf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vectorizer_path: str
    vectorizer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality: TopicRepresentationQuality
    warnings: list[str]
    created_at: datetime


class TopicTermBackend(Protocol):
    """Injectable vectorizer/c-TF-IDF boundary; it never changes cluster labels."""

    name: str
    library_version: str

    @property
    def vocabulary(self) -> Sequence[str]: ...

    def fit(self, documents: Sequence[str]) -> None: ...

    def top_terms(self, topic_row: int, count: int) -> list[tuple[str, float]]: ...

    def dump_vectorizer(self, path: Path) -> None: ...

    def dump_matrix(self, path: Path) -> None: ...


TopicBackendFactory = Callable[[TopicRepresentationConfig, set[str]], TopicTermBackend]


class CTFIDFBackend:
    """Production BERTopic c-TF-IDF adapter without BERTopic clustering."""

    name = "bertopic-ctfidf"

    def __init__(self, config: TopicRepresentationConfig, stopwords: set[str]) -> None:
        try:
            from bertopic.vectorizers import ClassTfidfTransformer
            from sklearn.feature_extraction.text import CountVectorizer
        except ImportError as exc:  # pragma: no cover - optional ML environment
            msg = "bertopic and scikit-learn are required; install the 'analysis' extra"
            raise RuntimeError(msg) from exc
        token_pattern = rf"(?u)\b[\w-]{{{config.minimum_token_length},}}\b"
        self._vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=token_pattern,
            stop_words=sorted(stopwords) or None,
            min_df=config.min_df,
            max_df=config.max_df,
            ngram_range=(config.ngram_min, config.ngram_max),
            max_features=config.max_features,
        )
        self._transformer = ClassTfidfTransformer(
            bm25_weighting=config.bm25_weighting,
            reduce_frequent_words=config.reduce_frequent_words,
        )
        self._matrix = None
        self._vocabulary: list[str] = []
        self.library_version = version("bertopic")

    @property
    def vocabulary(self) -> Sequence[str]:
        return self._vocabulary

    def fit(self, documents: Sequence[str]) -> None:
        from scipy.sparse import csr_matrix

        if not documents:
            self._matrix = csr_matrix((0, 0), dtype="float64")
            self._vocabulary = []
            return
        if len(documents) == 1:
            self._vectorizer.set_params(max_df=1.0)
        try:
            counts = self._vectorizer.fit_transform(documents)
        except ValueError as exc:
            msg = "topic vocabulary is empty or incompatible with min_df/max_df"
            raise ValueError(msg) from exc
        self._vocabulary = [str(value) for value in self._vectorizer.get_feature_names_out()]
        self._matrix = self._transformer.fit_transform(counts)

    def top_terms(self, topic_row: int, count: int) -> list[tuple[str, float]]:
        if self._matrix is None:
            msg = "topic backend must be fitted before reading terms"
            raise RuntimeError(msg)
        row = self._matrix.getrow(topic_row)
        candidates = [
            (self._vocabulary[int(index)], float(weight))
            for index, weight in zip(row.indices, row.data, strict=True)
            if weight > 0
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates[:count]

    def dump_vectorizer(self, path: Path) -> None:
        with path.open("wb") as target:
            pickle.dump(self._vectorizer, target, protocol=pickle.HIGHEST_PROTOCOL)

    def dump_matrix(self, path: Path) -> None:
        from scipy.sparse import save_npz

        if self._matrix is None:
            msg = "topic backend must be fitted before saving"
            raise RuntimeError(msg)
        with path.open("wb") as target:
            save_npz(target, self._matrix, compressed=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stopwords(config: TopicRepresentationConfig) -> set[str]:
    """Load configured language lists from the optional, versioned stopwords package."""
    words = set(config.extra_stopwords)
    if not config.stopword_languages:
        return words
    try:
        import stopwordsiso
    except ImportError as exc:  # pragma: no cover - optional ML environment
        msg = "stopwordsiso is required for configured stopword languages; install the 'analysis' extra"
        raise RuntimeError(msg) from exc
    for language in config.stopword_languages:
        language_words = stopwordsiso.stopwords(language)
        if not language_words:
            msg = f"stopwordsiso has no stopword list for language {language!r}"
            raise ValueError(msg)
        words.update(str(word).casefold() for word in language_words)
    return words


def _stopwords_sha256(stopwords: set[str]) -> str:
    payload = "\n".join(sorted(stopwords)).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_contracts(
    corpus_path: Path,
    embeddings_path: Path,
    labels_path: Path,
    probabilities_path: Path,
    corpus_manifest_path: Path,
    clustering_manifest_path: Path,
) -> tuple[CorpusManifest, ClusteringManifest]:
    corpus = CorpusManifest.model_validate_json(corpus_manifest_path.read_text(encoding="utf-8"))
    clustering = ClusteringManifest.model_validate_json(clustering_manifest_path.read_text(encoding="utf-8"))
    checks = (
        (_sha256_file(corpus_path), corpus.corpus_sha256, "corpus"),
        (_sha256_file(embeddings_path), corpus.final_embeddings_sha256, "embeddings"),
        (_sha256_file(labels_path), clustering.labels_sha256, "labels"),
        (_sha256_file(probabilities_path), clustering.probabilities_sha256, "probabilities"),
        (_sha256_file(corpus_manifest_path), clustering.corpus_manifest_sha256, "corpus manifest"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            msg = f"{label} checksum does not match its manifest"
            raise ValueError(msg)
    if clustering.corpus_sha256 != corpus.corpus_sha256:
        msg = "clustering and corpus manifests refer to different corpora"
        raise ValueError(msg)
    return corpus, clustering


def _load_assignments(
    labels_path: Path,
    probabilities_path: Path,
    records: int,
) -> tuple[NDArray[np.int64], NDArray[np.float32], int]:
    import numpy as np

    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    probabilities = np.load(probabilities_path, mmap_mode="r", allow_pickle=False)
    if labels.shape != (records,) or labels.dtype.kind not in "iu":
        msg = f"cluster labels must be an integer vector with {records} rows"
        raise ValueError(msg)
    if probabilities.shape != (records,):
        msg = f"cluster probabilities must contain {records} rows"
        raise ValueError(msg)
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        msg = "cluster probabilities must be finite values in [0, 1]"
        raise ValueError(msg)
    topic_ids = sorted({int(label) for label in labels if label >= 0})
    if topic_ids != list(range(len(topic_ids))) or np.any(labels < -1):
        msg = "cluster labels must use normalized contiguous topic IDs and -1 outliers"
        raise ValueError(msg)
    return np.asarray(labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float32), len(topic_ids)


@dataclass(slots=True)
class _TopicState:
    text_parts: list[str] = field(default_factory=list)
    characters: int = 0
    truncated: bool = False
    records: int = 0
    probability_sum: float = 0.0
    languages: Counter[str] = field(default_factory=Counter)
    videos: set[str] = field(default_factory=set)


def _aggregate_documents(
    corpus_path: Path,
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float32],
    topics: int,
    config: TopicRepresentationConfig,
) -> tuple[list[str], list[_TopicState]]:
    states = [_TopicState() for _ in range(topics)]
    count = 0
    with corpus_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if count >= len(labels):
                break
            record = CorpusRecord.model_validate_json(line)
            topic_id = int(labels[count])
            if topic_id >= 0:
                state = states[topic_id]
                state.records += 1
                state.probability_sum += float(probabilities[count])
                state.languages[record.detected_language] += 1
                if record.video_id:
                    state.videos.add(record.video_id)
                available = (
                    None
                    if config.max_cluster_characters is None
                    else config.max_cluster_characters - state.characters
                )
                if available is None or available > 0:
                    selected = record.clean_text if available is None else record.clean_text[:available]
                    if selected:
                        state.text_parts.append(selected)
                        state.characters += len(selected)
                    if available is not None and len(selected) < len(record.clean_text):
                        state.truncated = True
                else:
                    state.truncated = True
            count += 1
    if count != len(labels):
        msg = f"corpus supplied {count} records for {len(labels)} assignments"
        raise ValueError(msg)
    return ["\n".join(state.text_parts) for state in states], states


def _centroids(embeddings: NDArray[np.float32], labels: NDArray[np.int64], topics: int) -> NDArray[np.float32]:
    import numpy as np

    centroids = np.zeros((topics, embeddings.shape[1]), dtype=np.float64)
    counts = np.zeros(topics, dtype=np.int64)
    for start in range(0, len(labels), 8192):
        stop = min(len(labels), start + 8192)
        chunk_labels = labels[start:stop]
        chunk_vectors = embeddings[start:stop]
        for topic_id in range(topics):
            selected = chunk_vectors[chunk_labels == topic_id]
            if len(selected):
                centroids[topic_id] += np.sum(selected, axis=0)
                counts[topic_id] += len(selected)
    if topics and np.any(counts == 0):
        msg = "normalized labels contain an empty topic"
        raise ValueError(msg)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    if topics and np.any(norms == 0):
        msg = "a topic centroid is a zero vector"
        raise ValueError(msg)
    return np.asarray(centroids / norms, dtype=np.float32)


def _representatives(
    corpus_path: Path,
    embeddings: NDArray[np.float32],
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float32],
    centroids: NDArray[np.float32],
    config: TopicRepresentationConfig,
) -> list[RepresentativeIndices]:
    candidate_count = config.representatives_per_topic * config.representative_candidate_multiplier
    heaps: list[list[tuple[float, int, float, str, str]]] = [[] for _ in range(len(centroids))]
    count = 0
    with corpus_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            if count >= len(labels):
                break
            topic_id = int(labels[count])
            if topic_id >= 0 and probabilities[count] >= config.minimum_representative_probability:
                record = CorpusRecord.model_validate_json(line)
                vector = embeddings[count]
                norm = float((vector @ vector) ** 0.5)
                similarity = float((vector @ centroids[topic_id]) / norm) if norm else -1.0
                score = (float(probabilities[count]) + similarity) / 2
                candidate = (score, -count, similarity, record.video_id, record.author)
                if len(heaps[topic_id]) < candidate_count:
                    heapq.heappush(heaps[topic_id], candidate)
                elif candidate > heaps[topic_id][0]:
                    heapq.heapreplace(heaps[topic_id], candidate)
            count += 1
    results = []
    for topic_id, heap in enumerate(heaps):
        candidates = sorted(heap, reverse=True)
        selected: list[tuple[float, int, float, str, str]] = []
        used_videos: set[str] = set()
        used_authors: set[str] = set()
        for candidate in candidates:
            video, author = candidate[3], candidate[4]
            if (video and video in used_videos) or (author and author in used_authors):
                continue
            selected.append(candidate)
            used_videos.add(video)
            used_authors.add(author)
            if len(selected) == config.representatives_per_topic:
                break
        if len(selected) < config.representatives_per_topic:
            selected_indices = {-candidate[1] for candidate in selected}
            selected.extend(
                candidate
                for candidate in candidates
                if -candidate[1] not in selected_indices
            )
            selected = selected[: config.representatives_per_topic]
        results.append(
            RepresentativeIndices(
                topic_id=topic_id,
                record_indices=[-candidate[1] for candidate in selected],
                centroid_similarities=[candidate[2] for candidate in selected],
            ),
        )
    return results


def _keyword_kind(term: str) -> str:
    return "ngram" if " " in term else "unigram"


def _topic_quality(
    keywords: list[list[TopicKeyword]],
    names: list[str],
    representatives: list[RepresentativeIndices],
    vocabulary_size: int,
    config: TopicRepresentationConfig,
) -> TopicRepresentationQuality:
    term_sets = [{keyword.term for keyword in topic} for topic in keywords]
    total_slots = sum(len(topic) for topic in keywords)
    unique_terms = len(set().union(*term_sets)) if term_sets else 0
    maximum_jaccard = 0.0
    similar_pairs = 0
    for left in range(len(term_sets)):
        for right in range(left + 1, len(term_sets)):
            union = term_sets[left] | term_sets[right]
            score = len(term_sets[left] & term_sets[right]) / len(union) if union else 0.0
            maximum_jaccard = max(maximum_jaccard, score)
            if score >= config.similar_topic_jaccard_warning:
                similar_pairs += 1
    similarities = [value for item in representatives for value in item.centroid_similarities]
    duplicate_names = sum(count - 1 for count in Counter(names).values() if count > 1)
    return TopicRepresentationQuality(
        topics=len(keywords),
        empty_topics=sum(not topic for topic in keywords),
        vocabulary_size=vocabulary_size,
        topic_diversity=unique_terms / total_slots if total_slots else 0,
        maximum_keyword_jaccard=maximum_jaccard,
        similar_topic_pairs=similar_pairs,
        duplicate_names=duplicate_names,
        mean_representative_similarity=sum(similarities) / len(similarities) if similarities else None,
    )


def _ensure_labels_unchanged(path: Path, expected_sha256: str) -> None:
    if _sha256_file(path) != expected_sha256:
        msg = "cluster labels changed while building topic representations"
        raise ValueError(msg)


def build_topic_representations(
    corpus_path: Path,
    embeddings_path: Path,
    labels_path: Path,
    probabilities_path: Path,
    corpus_manifest_path: Path,
    clustering_manifest_path: Path,
    output_dir: Path,
    *,
    config: TopicRepresentationConfig | None = None,
    backend_factory: TopicBackendFactory | None = None,
    stopwords: set[str] | None = None,
    force: bool = False,
    limit_topics: int | None = None,
) -> TopicRepresentationManifest:
    """Build c-TF-IDF topics without changing HDBSCAN assignments or outliers."""
    import numpy as np

    active_config = config or TopicRepresentationConfig()
    corpus, clustering = _validate_contracts(
        corpus_path,
        embeddings_path,
        labels_path,
        probabilities_path,
        corpus_manifest_path,
        clustering_manifest_path,
    )
    records = clustering.output_records
    labels, probabilities, source_topics = _load_assignments(labels_path, probabilities_path, records)
    if limit_topics is not None and limit_topics < 1:
        msg = "limit_topics must be at least 1"
        raise ValueError(msg)
    topics = source_topics if limit_topics is None else min(limit_topics, source_topics)
    working_labels = np.asarray(labels.copy())
    working_labels[working_labels >= topics] = -1
    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    if embeddings.shape != (corpus.stats.output_records, corpus.dimensions) or embeddings.dtype != np.float32:
        msg = "final embeddings shape or dtype does not match corpus manifest"
        raise ValueError(msg)
    active_stopwords = set(stopwords) if stopwords is not None else load_stopwords(active_config)
    factory = backend_factory or CTFIDFBackend

    output_dir.mkdir(parents=True, exist_ok=True)
    representations_path = output_dir / "topic-representations.jsonl"
    keywords_path = output_dir / "topic-keywords.jsonl"
    representative_path = output_dir / "representative-indices.jsonl"
    vocabulary_path = output_dir / "vocabulary.json"
    ctfidf_path = output_dir / "ctfidf.npz"
    vectorizer_path = output_dir / "vectorizer.pkl"
    manifest_path = output_dir / "topic-representation-manifest.json"
    report_path = output_dir / "topic-representation-report.md"
    finals = (
        representations_path,
        keywords_path,
        representative_path,
        vocabulary_path,
        ctfidf_path,
        vectorizer_path,
        manifest_path,
        report_path,
    )
    existing = next((path for path in finals if path.exists()), None)
    if existing is not None and not force:
        msg = f"refusing to overwrite existing topic-representation artifact: {existing}"
        raise FileExistsError(msg)

    documents, states = _aggregate_documents(corpus_path, working_labels, probabilities, topics, active_config)
    backend = factory(active_config, active_stopwords)
    backend.fit(documents)
    keyword_lists = [
        [
            TopicKeyword(term=term, weight=weight, rank=rank, kind=_keyword_kind(term))
            for rank, (term, weight) in enumerate(backend.top_terms(topic_id, active_config.top_n_words), start=1)
        ]
        for topic_id in range(topics)
    ]
    names = [" / ".join(keyword.term for keyword in keywords[:3]) for keywords in keyword_lists]
    centroids = _centroids(embeddings[:records], working_labels, topics)
    representatives = _representatives(
        corpus_path,
        embeddings,
        working_labels,
        probabilities,
        centroids,
        active_config,
    )
    representations = [
        TopicRepresentation(
            topic_id=topic_id,
            name=names[topic_id],
            records=states[topic_id].records,
            mean_probability=states[topic_id].probability_sum / states[topic_id].records,
            languages=dict(states[topic_id].languages),
            unique_videos=len(states[topic_id].videos),
            keywords=keyword_lists[topic_id],
            representative_indices=representatives[topic_id].record_indices,
        )
        for topic_id in range(topics)
    ]
    quality = _topic_quality(keyword_lists, names, representatives, len(backend.vocabulary), active_config)
    warnings = []
    if quality.empty_topics:
        warnings.append(f"topics without informative terms: {quality.empty_topics}")
    if quality.similar_topic_pairs:
        warnings.append(f"highly similar topic pairs: {quality.similar_topic_pairs}")
    if quality.duplicate_names:
        warnings.append(f"duplicate generated topic names: {quality.duplicate_names}")
    truncated_topics = sum(state.truncated for state in states)
    if truncated_topics:
        warnings.append(f"cluster documents truncated by character limit: {truncated_topics}")

    temporary = {path: path.with_name(f".{path.name}.tmp") for path in finals}
    for path in temporary.values():
        path.unlink(missing_ok=True)
    try:
        with temporary[representations_path].open("w", encoding="utf-8") as target:
            for representation in representations:
                target.write(f"{representation.model_dump_json()}\n")
        with temporary[keywords_path].open("w", encoding="utf-8") as target:
            for topic_id, keywords in enumerate(keyword_lists):
                payload = {"topic_id": topic_id, "keywords": [item.model_dump() for item in keywords]}
                target.write(f"{json.dumps(payload)}\n")
        with temporary[representative_path].open("w", encoding="utf-8") as target:
            for representative in representatives:
                target.write(f"{representative.model_dump_json()}\n")
        temporary[vocabulary_path].write_text(
            f"{json.dumps(list(backend.vocabulary), ensure_ascii=False, indent=2)}\n",
            encoding="utf-8",
        )
        backend.dump_matrix(temporary[ctfidf_path])
        backend.dump_vectorizer(temporary[vectorizer_path])
        _ensure_labels_unchanged(labels_path, clustering.labels_sha256)
        manifest = TopicRepresentationManifest(
            corpus_manifest_path=str(corpus_manifest_path),
            corpus_manifest_sha256=_sha256_file(corpus_manifest_path),
            clustering_manifest_path=str(clustering_manifest_path),
            clustering_manifest_sha256=_sha256_file(clustering_manifest_path),
            corpus_path=str(corpus_path),
            corpus_sha256=corpus.corpus_sha256,
            embeddings_path=str(embeddings_path),
            embeddings_sha256=corpus.final_embeddings_sha256,
            labels_path=str(labels_path),
            labels_sha256=clustering.labels_sha256,
            probabilities_path=str(probabilities_path),
            probabilities_sha256=clustering.probabilities_sha256,
            input_records=records,
            source_topics=source_topics,
            topics=topics,
            omitted_topics=source_topics - topics,
            outliers=int(np.count_nonzero(labels == -1)),
            config=active_config,
            stopwords_sha256=_stopwords_sha256(active_stopwords),
            backend=backend.name,
            backend_version=backend.library_version,
            representations_path=str(representations_path),
            representations_sha256=_sha256_file(temporary[representations_path]),
            keywords_path=str(keywords_path),
            keywords_sha256=_sha256_file(temporary[keywords_path]),
            representative_indices_path=str(representative_path),
            representative_indices_sha256=_sha256_file(temporary[representative_path]),
            vocabulary_path=str(vocabulary_path),
            vocabulary_sha256=_sha256_file(temporary[vocabulary_path]),
            ctfidf_path=str(ctfidf_path),
            ctfidf_sha256=_sha256_file(temporary[ctfidf_path]),
            vectorizer_path=str(vectorizer_path),
            vectorizer_sha256=_sha256_file(temporary[vectorizer_path]),
            quality=quality,
            warnings=warnings,
            created_at=datetime.now(UTC),
        )
        temporary[manifest_path].write_text(f"{manifest.model_dump_json(indent=2)}\n", encoding="utf-8")
        temporary[report_path].write_text(_topic_report(manifest), encoding="utf-8")
        for final in (*finals[:-2], report_path, manifest_path):
            temporary[final].replace(final)
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    return manifest


def _topic_report(manifest: TopicRepresentationManifest) -> str:
    quality = manifest.quality
    warnings = "\n".join(f"- {warning}" for warning in manifest.warnings) or "- None."
    return f"""# Topic representations

- Input records: {manifest.input_records}
- Topics: {manifest.topics}
- Omitted topics in trial: {manifest.omitted_topics}
- Preserved outliers: {manifest.outliers}
- Vocabulary size: {quality.vocabulary_size}
- Empty topics: {quality.empty_topics}
- Topic diversity: {quality.topic_diversity:.4f}
- Maximum keyword Jaccard: {quality.maximum_keyword_jaccard:.4f}
- Similar topic pairs: {quality.similar_topic_pairs}
- Duplicate generated names: {quality.duplicate_names}
- Mean representative similarity: {quality.mean_representative_similarity}
- Backend: `{manifest.backend}=={manifest.backend_version}`

## Warnings

{warnings}

HDBSCAN labels and outliers are unchanged. The persisted vectorizer is a local trusted pickle and must not be loaded
without checksum and origin verification. Reports contain aggregates only; representative artifacts contain indexes,
not comment text.
"""
