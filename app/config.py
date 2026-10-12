"""Central settings, read once from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional at runtime
    pass

ROOT = Path(__file__).resolve().parent.parent


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("BANKMATE_MODEL", "gemini-2.5-flash")
    embed_model: str = os.getenv("BANKMATE_EMBED_MODEL", "gemini-embedding-001")
    api_key_present: bool = bool(os.getenv("GOOGLE_API_KEY"))
    offline_mode: bool = _flag("OFFLINE_MODE") or not bool(os.getenv("GOOGLE_API_KEY"))
    use_embeddings: bool = _flag("USE_EMBEDDINGS", "1") and bool(os.getenv("GOOGLE_API_KEY"))

    policies_dir: Path = ROOT / "data" / "policies"
    index_dir: Path = ROOT / "data" / "index"
    db_path: Path = Path(os.getenv("BANKMATE_DB", str(ROOT / "data" / "bank.db")))

    chunk_words: int = 120
    chunk_overlap: int = 30
    top_k: int = 4
    # Below this BM25 score (BM25-only mode), results are treated as "not found"
    min_bm25_score: float = 1.0

    app_name: str = "bankmate"
    max_input_chars: int = 2000
    allowed_origins: list[str] = field(default_factory=lambda: ["*"])


settings = Settings()
