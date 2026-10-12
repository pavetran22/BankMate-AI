"""Hybrid retriever: BM25 keyword search + Gemini embeddings, fused with
Reciprocal Rank Fusion (RRF).

* BM25 is always on. It is fast, needs no API, and is strong on exact terms
  such as "RM250", "IBG", "CCRIS".
* Dense embeddings (gemini-embedding-001) are added when an API key is set.
  They catch paraphrases such as "how much do I pay if I'm late" vs
  "Late Payment Charge".
* RRF merges the two ranked lists without having to calibrate their scores.

Embeddings are cached on disk keyed by a content hash, so re-indexing only
embeds chunks that changed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.config import settings
from app.rag.chunking import Chunk, chunk_directory

log = logging.getLogger(__name__)

_STOPWORDS = set(
    """a an and are as at be by can do does for from has have how i if in is it
    its me my of on or the to what when where which who will with you your
    there this that than then much many any""".split()
)
RRF_K = 60


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


@dataclass
class SearchHit:
    chunk: Chunk
    score: float          # fused score (RRF) or BM25 score in BM25-only mode
    bm25: float
    dense: float | None   # cosine similarity, None when embeddings are off

    def to_dict(self) -> dict:
        return {
            "citation": self.chunk.citation,
            "document": self.chunk.doc_title,
            "section": self.chunk.section,
            "text": self.chunk.text,
            "score": round(self.score, 4),
        }


class Embedder:
    """Thin wrapper around the Gemini embedding API with an on-disk cache."""

    def __init__(self, model: str, cache_path):
        from google import genai  # imported lazily so BM25-only mode needs no key

        self.client = genai.Client()
        self.model = model
        self.cache_path = cache_path
        self.cache: dict[str, list[float]] = {}
        if cache_path.exists():
            self.cache = json.loads(cache_path.read_text())

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), 50):  # API batch limit safety
            resp = self.client.models.embed_content(
                model=self.model,
                contents=texts[i : i + 50],
                config=types.EmbedContentConfig(task_type=task_type),
            )
            out.extend(e.values for e in resp.embeddings)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha256(f"{self.model}|{t}".encode()).hexdigest() for t in texts]
        missing = [(k, t) for k, t in zip(keys, texts) if k not in self.cache]
        if missing:
            log.info("Embedding %d new chunks", len(missing))
            vectors = self._embed([t for _, t in missing], "RETRIEVAL_DOCUMENT")
            for (k, _), v in zip(missing, vectors):
                self.cache[k] = v
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))
        return [self.cache[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None):
        if not chunks:
            raise ValueError("No policy chunks to index - check data/policies/")
        self.chunks = chunks
        # Index heading + title with the body so section names are searchable
        self._indexed_text = [f"{c.doc_title}. {c.section}. {c.text}" for c in chunks]
        self.bm25 = BM25Okapi([tokenize(t) for t in self._indexed_text])
        self.embedder = embedder
        self.vectors = embedder.embed_documents(self._indexed_text) if embedder else None

    @property
    def mode(self) -> str:
        return "hybrid (BM25 + Gemini embeddings)" if self.embedder else "BM25 only"

    def search(self, query: str, k: int | None = None) -> list[SearchHit]:
        k = k or settings.top_k
        bm25_scores = [float(s) for s in self.bm25.get_scores(tokenize(query))]
        bm25_rank = sorted(range(len(self.chunks)), key=lambda i: -bm25_scores[i])

        qv = None
        if self.vectors:
            try:
                qv = self.embedder.embed_query(query)
            except Exception as exc:  # API hiccup: degrade to keyword search, don't fail
                log.warning("Query embedding failed (%s); BM25 only for this query", exc)

        if qv is None:
            return [
                SearchHit(self.chunks[i], bm25_scores[i], bm25_scores[i], None)
                for i in bm25_rank[:k]
                if bm25_scores[i] >= settings.min_bm25_score
            ]

        dense_scores = [_cosine(qv, v) for v in self.vectors]
        dense_rank = sorted(range(len(self.chunks)), key=lambda i: -dense_scores[i])

        fused: dict[int, float] = {}
        for ranking in (bm25_rank, dense_rank):
            for pos, idx in enumerate(ranking[:20]):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + pos + 1)

        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [
            SearchHit(self.chunks[i], fused[i], bm25_scores[i], dense_scores[i])
            for i in best
            # Drop results that neither retriever thinks are relevant
            if bm25_scores[i] >= settings.min_bm25_score or dense_scores[i] >= 0.55
        ]


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    chunks = chunk_directory(settings.policies_dir, settings.chunk_words, settings.chunk_overlap)
    embedder = None
    if settings.use_embeddings and not settings.offline_mode:
        try:
            embedder = Embedder(settings.embed_model, settings.index_dir / "embeddings.json")
        except Exception as exc:  # fall back gracefully, keep the app usable
            log.warning("Embeddings unavailable (%s); using BM25 only", exc)
    try:
        retriever = HybridRetriever(chunks, embedder)
    except Exception as exc:
        if embedder is None:
            raise
        log.warning("Embedding index failed (%s); using BM25 only", exc)
        retriever = HybridRetriever(chunks, None)
    log.info("Retriever ready: %d chunks, mode=%s", len(chunks), retriever.mode)
    return retriever
