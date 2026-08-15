#!/usr/bin/env python3
"""Offline pain-mining: comments JSONL -> embeddings -> near-dup -> BERTopic clusters.

Reads a JSONL comment export, cleans it, embeds with a local multilingual-e5 model,
drops semantic near-duplicates by a cosine threshold, clusters with BERTopic, reassigns
only confidently on-topic HDBSCAN outliers (thresholded embedding similarity; rest stay -1), and
writes machine-readable results to --out-dir (see write_results for the schema). No
LLM/network calls beyond the one-time model download.

Usage:
    python scripts/mine_topics.py comments.jsonl --threads 16 --min-topic-size 250
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from pydantic import ValidationError

# Running a file inside scripts/ puts that directory, rather than the repository root, on
# sys.path. Add the root so this standalone CLI can reuse the application's noise filter.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.config import (
    AnalysisConfig,
    CleaningConfig,
    ClusteringConfig,
    DeduplicationConfig,
    EmbeddingConfig,
)
from src.analysis.models import CommentRecord
from src.analysis.schemas import ANALYSIS_SCHEMA_VERSION, ClusterComment, ClusterRecord, ExportedComment
from src.infrastructure.extractor.noise import is_noise as is_pipeline_noise

if TYPE_CHECKING:
    import numpy as np


def limit_threads(n_threads: int) -> None:
    """Cap CPU parallelism. Must run before torch/numpy/numba import (they read these at import)."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(var, str(n_threads))


# Short acknowledgements that form dense but useless clusters (valid Russian, so gates let them through).
THANKS_TOKENS = {
    "спс", "спасибо", "благодарю", "класс", "супер", "круто", "лайк",
    "огонь", "топ", "красота", "молодец", "молодцы", "здорово", "отлично", "браво",
}

# Russian stopwords for c-TF-IDF keyword extraction (sklearn ships none).
# stopwordsiso ru = 559 words, bundled with the package (no runtime download, unlike NLTK).
# Fold in THANKS_TOKENS so bare acknowledgements never surface as cluster keywords.
def _russian_stopwords() -> list[str]:
    import stopwordsiso

    return sorted(stopwordsiso.stopwords("ru") | THANKS_TOKENS)

WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)
CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
LATIN_RE = re.compile(r"[a-z]", re.IGNORECASE)


def load_comments(path: Path) -> list[CommentRecord]:
    """Read a JSONL export, keeping text + domain facets (author/channel/query/video_id)."""
    rows: list[CommentRecord] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            comment = CommentRecord.from_export(ExportedComment.model_validate(json.loads(line)))
            if comment.text:
                rows.append(comment)
    return rows


def is_noise(text: str, min_length: int) -> bool:
    """Drop too-short comments, bare thanks/emoji, and predominantly non-Russian text."""
    stripped = text.strip()
    if len(stripped) < min_length:
        return True
    words = WORD_RE.findall(stripped.lower())
    if not words:
        return True
    if len(words) <= 3 and all(w in THANKS_TOKENS for w in words):
        return True
    n_cyr = len(CYRILLIC_RE.findall(stripped))
    n_lat = len(LATIN_RE.findall(stripped))
    return n_cyr + n_lat > 0 and n_cyr / (n_cyr + n_lat) < 0.5


def clean(rows: list[CommentRecord], min_length: int, spam_filter: bool = False) -> list[CommentRecord]:
    """Filter noise and drop exact-duplicate texts (near-dup handled later on embeddings).

    With spam_filter=True, additionally apply the pipeline's cheap noise heuristics —
    drops glued gibberish and emoji/symbol spam before embedding.
    Off by default so the calibrated runs in the methodology (§9) stay reproducible.
    """
    seen: set[str] = set()
    kept: list[CommentRecord] = []
    for row in rows:
        if is_noise(row.text, min_length):
            continue
        if spam_filter and is_pipeline_noise(row.text):
            continue
        key = row.normalized_text_key
        if key in seen:
            continue
        seen.add(key)
        kept.append(row)
    return kept


def clustering_prefix(model_name: str) -> str:
    """Return the input prefix this model expects for a clustering/symmetric use-case.

    Embedding models bake a prompt convention into training; feeding the wrong one (or none)
    measurably degrades quality, so the prefix must match the model — not be hardcoded. Refs:
      - e5 (multilingual-e5-*): "query: " for symmetric tasks incl. clustering (model card).
      - FRIDA (ai-forever/FRIDA): task prefixes; "categorize_topic: " is the topic-grouping one.
      - BGE-M3 / USER-bge-m3 (deepvk): no prefix for symmetric tasks (BGE-M3 convention).
    Unknown models default to no prefix (safest) with a heads-up on stderr.
    """
    name = model_name.lower()
    if "e5" in name:
        return "query: "
    if "frida" in name:
        return "categorize_topic: "
    if "bge-m3" in name or "user-bge" in name or "bge_m3" in name:
        return ""
    print(f"WARNING: unknown model {model_name!r}; using no prefix. Check its card for a required prompt.",
          file=sys.stderr)
    return ""


def embed(texts: list[str], model_name: str, n_threads: int, batch_size: int = 0, max_seq_length: int = 512) -> np.ndarray:
    """Embed texts with a local sentence-transformers model. Uses a CUDA GPU if present."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(n_threads)
    print(f"Loading embedding model: {model_name} (threads={n_threads}) ...", file=sys.stderr)
    model = SentenceTransformer(model_name)
    # Cap sequence length: comments are short, so 512 tokens is plenty. e5 already defaults to 512,
    # but BGE-M3-family models default to 8192 — at a large batch that alone OOMs a 16GB GPU.
    model.max_seq_length = min(model.max_seq_length, max_seq_length)
    prefix = clustering_prefix(model_name)
    prefixed = [f"{prefix}{t}" for t in texts]  # model-specific prompt (see clustering_prefix)
    # Default 64 on GPU is safe across model sizes (BGE-M3-large is heavier than e5); raise via
    # --batch-size if you have headroom. CPU: batch size barely affects speed, just memory.
    if batch_size <= 0:
        batch_size = 64
    print(f"Embedding {len(texts)} comments (batch_size={batch_size}, max_seq={model.max_seq_length}, "
          f"prefix={prefix!r}) ...", file=sys.stderr)
    vectors = model.encode(prefixed, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vectors, dtype=np.float32)


def near_dedup(
    embeddings: np.ndarray,
    threshold: float,
    block_size: int = 2048,
    n_examples: int = 8,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Collapse near-duplicates (cosine >= threshold) into one representative each.

    Embeddings are L2-normalized, so dot product == cosine. Streamed in row-blocks to avoid the
    full n*n matrix. Threshold is model-specific — eyeball the returned sample pairs to re-tune.
    Returns (indices_to_keep, sample (representative, dropped) index pairs).
    """
    import numpy as np

    n = embeddings.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        lo, hi = (ra, rb) if ra < rb else (rb, ra)
        parent[hi] = lo  # earliest index is the representative

    print(f"Near-dup: scanning {n} vectors (cosine >= {threshold}) ...", file=sys.stderr)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        sims = embeddings[start:end] @ embeddings.T
        for r in range(end - start):
            gi = start + r
            for offset in np.nonzero(sims[r, gi + 1 :] > threshold)[0]:  # forward-only: matrix is symmetric
                union(gi, gi + 1 + int(offset))

    keep = [i for i in range(n) if find(i) == i]
    examples: list[tuple[int, int]] = []
    for i in range(n):
        rep = find(i)
        if rep != i and len(examples) < n_examples:
            examples.append((rep, i))
    return keep, examples


def build_topic_model(min_topic_size: int) -> object:
    """BERTopic: UMAP + HDBSCAN + Russian-aware BM25 c-TF-IDF, stable seed.

    Keywords come from plain c-TF-IDF. We deliberately do NOT add KeyBERTInspired here: on this
    corpus it reranks by similarity to a cluster's representative docs, which on the noisier,
    conversational clusters promotes filler/question words ("как", "ты", "же") over the rarer
    distinctive terms that c-TF-IDF surfaces. Plain c-TF-IDF gives cleaner topic labels here.
    """
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    vectorizer = CountVectorizer(
        stop_words=_russian_stopwords(),
        ngram_range=(1, 2),
        min_df=1,  # BERTopic fits this per-topic, so min_df counts topics not comments
        token_pattern=r"(?u)\b[а-яё][а-яё-]+\b",  # Cyrillic-only keywords
    )
    # BM25 weighting + downweighting frequent words: keeps generic chatter (e.g. "видео", "канал")
    # from dominating every topic's keywords instead of just the topic-specific ones.
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="leaf",  # 'eom' collapses everything into one topic here
        prediction_data=True,
    )
    return BERTopic(
        # Default "english" strips non-ASCII in c-TF-IDF preprocessing, wiping Cyrillic
        # and crashing on empty vocabulary; "multilingual" keeps the text intact.
        language="multilingual",
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        verbose=True,
    )


def write_results(
    topic_model: object,
    comments: list[CommentRecord],
    topics: list[int],
    out_dir: Path,
    top_n: int,
) -> tuple[int, int]:
    """Write one self-contained clusters.jsonl (+ human summary). Returns (n_topics, n_outliers).

    clusters.jsonl: one line per discovered cluster, largest first, outliers excluded:
      {topic_id, n_comments, n_authors, n_channels, keywords[],
       comments: [{text, author, channel, query, video_id, video_title, video_url}, ...]}
    Everything to render a cluster (its comments + where each came from) is in that one record —
    no joins. topics_summary.md is a truncated human-readable overview for eyeballing the run.
    """
    texts = [comment.text for comment in comments]
    info = topic_model.get_topic_info()
    n_topics = int((info["Topic"] >= 0).sum())
    n_outliers = sum(1 for t in topics if t == -1)

    # Group by topic, keeping full comment records; distinct authors/channels = honest breadth.
    authors_by_topic: dict[int, set[str]] = {}
    channels_by_topic: dict[int, set[str]] = {}
    comments_by_topic: dict[int, list[ClusterComment]] = {}
    for comment, topic in zip(comments, topics, strict=True):
        authors_by_topic.setdefault(topic, set()).add(comment.author)
        channels_by_topic.setdefault(topic, set()).add(comment.channel)
        comments_by_topic.setdefault(topic, []).append(comment.to_cluster_comment())

    out_dir.mkdir(parents=True, exist_ok=True)
    md: list[str] = ["# Discovered topics (BERTopic)\n"]

    with (out_dir / "clusters.jsonl").open("w", encoding="utf-8") as fh:
        for _, row in info.iterrows():
            topic_id = int(row["Topic"])
            if topic_id == -1:
                continue
            keywords = [w for w, _ in topic_model.get_topic(topic_id)[:8] if w]  # drop "" from degenerate topics
            topic_comments = comments_by_topic.get(topic_id, [])
            cluster = ClusterRecord(
                topic_id=topic_id,
                n_comments=int(row["Count"]),
                n_authors=len(authors_by_topic.get(topic_id, set())),
                n_channels=len(channels_by_topic.get(topic_id, set())),
                keywords=keywords,
                comments=topic_comments,
            )
            fh.write(
                json.dumps(cluster.model_dump(mode="json"), ensure_ascii=False)
                + "\n",
            )
            if topic_id < top_n:  # human summary: keywords + 5 sample texts
                md.append(f"\n## Topic {topic_id} — {int(row['Count'])} comments, "
                          f"{len(authors_by_topic.get(topic_id, set()))} authors, "
                          f"{len(channels_by_topic.get(topic_id, set()))} channels\n")
                md.append(f"**Keywords:** {', '.join(keywords)}\n")
                md.append("**Samples:**\n")
                md.extend(f"- {comment.text.replace(chr(10), ' ').strip()[:300]}" for comment in topic_comments[:5])

    (out_dir / "topics_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\nDiscovered {n_topics} topics; {n_outliers}/{len(topics)} comments are outliers.")
    print(f"Wrote {out_dir}/clusters.jsonl and topics_summary.md")
    return n_topics, n_outliers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover pain topics from a comments JSONL via BERTopic.")
    parser.add_argument("jsonl", type=Path, help="Path to the comments JSONL export.")
    parser.add_argument(
        "--model", default="intfloat/multilingual-e5-large",
        help="sentence-transformers model. Default e5-large consolidates a pain into one broad, "
        "high-channel-breadth topic (good for prioritising widespread pains). Alternative for finer "
        "niche discovery: deepvk/USER-bge-m3 (more, tighter topics) — but its cosines run lower, so "
        "pair it with --reduce-outliers-threshold ~0.7 (e5 uses 0.9).",
    )
    parser.add_argument("--min-topic-size", type=int, default=250, help="HDBSCAN min cluster size (~N/500 at scale).")
    parser.add_argument("--min-length", type=int, default=20, help="Drop comments shorter than this many chars.")
    parser.add_argument(
        "--spam-filter",
        action="store_true",
        help="Also apply the pipeline's emoji/symbol and glued-gibberish noise heuristics as an input gate.",
    )
    parser.add_argument("--near-dup-threshold", type=float, default=0.95, help="Cosine >= this = near-duplicate.")
    parser.add_argument("--no-near-dup", action="store_true", help="Skip the near-duplicate collapse.")
    parser.add_argument(
        "--no-reduce-outliers",
        action="store_true",
        help="Skip reassigning HDBSCAN outliers (-1) to a topic entirely (keep them all as -1).",
    )
    parser.add_argument(
        "--reduce-outliers-threshold",
        type=float,
        default=0.9,
        help="Only reassign an outlier if its embedding cosine similarity to a topic centroid clears "
        "this; else it stays -1. e5 cosines run high/compressed, so the selective band is ~0.88-0.92: "
        "at 0.9 on-topic outliers are recovered while chatter stays out; below ~0.85 it pulls in junk.",
    )
    parser.add_argument("--top-n", type=int, default=50, help="How many topics to write.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/analysis-output"), help="Where to write results.")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads cap (raise to core count on a server).")
    parser.add_argument("--batch-size", type=int, default=0, help="Embedding batch size (0 = auto=64). Lower on OOM.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N cleaned comments (0 = all).")
    parser.add_argument("--sample", type=int, default=0, help="Randomly sample N raw comments before cleaning (0 = all).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --sample (reproducible dev subsets).")
    return parser


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    """Translate the stable CLI interface into the strict internal configuration model."""
    return AnalysisConfig(
        input_path=args.jsonl,
        output_dir=args.out_dir,
        limit=None if args.limit == 0 else args.limit,
        sample_size=None if args.sample == 0 else args.sample,
        cleaning=CleaningConfig(min_length=args.min_length, spam_filter=args.spam_filter),
        embedding=EmbeddingConfig(
            model_name=args.model,
            threads=args.threads,
            batch_size=64 if args.batch_size == 0 else args.batch_size,
        ),
        deduplication=DeduplicationConfig(
            enabled=not args.no_near_dup,
            threshold=args.near_dup_threshold,
        ),
        clustering=ClusteringConfig(
            min_topic_size=args.min_topic_size,
            reduce_outliers=not args.no_reduce_outliers,
            reduce_outliers_threshold=args.reduce_outliers_threshold,
            random_seed=args.seed,
            top_n=args.top_n,
        ),
    )


def parse_config(argv: Sequence[str] | None = None) -> AnalysisConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return config_from_args(args)
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())
        parser.error(f"invalid analysis configuration: {details}")


def main() -> None:
    config = parse_config()

    limit_threads(config.embedding.threads)

    rows = load_comments(config.input_path)
    if config.sample_size is not None and config.sample_size < len(rows):
        import random

        random.Random(config.clustering.random_seed).shuffle(rows)
        rows = rows[: config.sample_size]
        print(
            f"Sampled {len(rows)} random comments (--sample, seed={config.clustering.random_seed}).",
            file=sys.stderr,
        )
    kept = clean(rows, config.cleaning.min_length, spam_filter=config.cleaning.spam_filter)
    gate = " (+spam-filter)" if config.cleaning.spam_filter else ""
    print(f"Loaded {len(rows)} comments; kept {len(kept)} after cleaning{gate}.", file=sys.stderr)

    if config.limit is not None:
        kept = kept[: config.limit]
        print(f"Limited to first {len(kept)} comments (--limit).", file=sys.stderr)

    n_authors = len(Counter(comment.author for comment in kept))
    print(
        f"Distinct authors: {n_authors}; distinct channels: {len({comment.channel for comment in kept})}.",
        file=sys.stderr,
    )

    texts = [comment.text for comment in kept]
    n_clean = len(texts)

    import numpy as np

    # Cache embeddings, keyed by model + a content hash (not just count: two runs can have the
    # same N from different --sample seeds or --min-length values, which would silently reuse
    # the wrong vectors if the count alone were the key).
    config.output_dir.mkdir(parents=True, exist_ok=True)
    content_hash = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:16]  # noqa: S324
    cache = config.output_dir / f"emb_{config.embedding.model_name.split('/')[-1]}_{len(texts)}_{content_hash}.npy"
    if cache.exists():
        print(f"Reusing cached embeddings: {cache}", file=sys.stderr)
        embeddings = np.load(cache)
    else:
        embeddings = embed(
            texts,
            config.embedding.model_name,
            config.embedding.threads,
            batch_size=config.embedding.batch_size,
            max_seq_length=config.embedding.max_seq_length,
        )
        np.save(cache, embeddings)

    if config.deduplication.enabled:
        keep_idx, examples = near_dedup(
            embeddings,
            config.deduplication.threshold,
            block_size=config.deduplication.block_size,
            n_examples=config.deduplication.sample_pairs,
        )
        print(f"Near-dup: kept {len(keep_idx)}, dropped {len(texts) - len(keep_idx)}.", file=sys.stderr)
        for rep, dup in examples:  # eyeball to re-tune threshold for this model
            print(f"  keep {texts[rep][:120]!r}\n  drop {texts[dup][:120]!r}", file=sys.stderr)
        embeddings = embeddings[np.array(keep_idx)]
        texts = [texts[i] for i in keep_idx]
        kept = [kept[i] for i in keep_idx]

    topic_model = build_topic_model(config.clustering.min_topic_size)
    print("Fitting BERTopic ...", file=sys.stderr)
    topics_arr, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    topics: list[int] = list(topics_arr)
    n_outliers_before = sum(1 for t in topics if t == -1)

    if config.clustering.reduce_outliers and n_outliers_before:
        # Thresholded embedding reassignment: an outlier joins the topic whose centroid it's closest
        # to, but only if that cosine similarity clears --reduce-outliers-threshold; otherwise it
        # honestly stays -1. We use the "embeddings" strategy (not "c-tf-idf") because these are short
        # comments: their c-TF-IDF vectors are too sparse to match any topic above a sane threshold
        # (c-TF-IDF reassigned ~zero here), whereas embeddings capture short-text meaning. Reuses the
        # vectors we already computed — no re-encoding.
        print(
            f"Reducing {n_outliers_before} outliers "
            f"(embeddings, threshold={config.clustering.reduce_outliers_threshold}) ...",
            file=sys.stderr,
        )
        topics = topic_model.reduce_outliers(
            texts, topics, strategy="embeddings", embeddings=embeddings,
            threshold=config.clustering.reduce_outliers_threshold,
        )
        # Refresh keywords with the new assignments. Must re-pass our vectorizer + ctfidf: without
        # them update_topics silently falls back to a DEFAULT CountVectorizer, dropping the Russian
        # stopwords and Cyrillic-only token pattern (letting "как"/"да" and even digits into keywords).
        topic_model.update_topics(
            texts, topics=topics,
            vectorizer_model=topic_model.vectorizer_model,
            ctfidf_model=topic_model.ctfidf_model,
        )

    n_topics, n_outliers = write_results(
        topic_model,
        kept,
        topics,
        config.output_dir,
        config.clustering.top_n,
    )

    run_meta = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "model": config.embedding.model_name,
        "near_dup_threshold": config.deduplication.threshold if config.deduplication.enabled else None,
        "reduce_outliers_threshold": (
            config.clustering.reduce_outliers_threshold if config.clustering.reduce_outliers else None
        ),
        "min_topic_size": config.clustering.min_topic_size,
        "n_input": len(rows),
        "n_after_clean": n_clean,
        "n_after_dedup": len(texts),
        "n_topics": n_topics,
        "n_outliers_before_reduction": n_outliers_before,
        "n_outliers": n_outliers,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (config.output_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {config.output_dir}/run_meta.json")


if __name__ == "__main__":
    main()
