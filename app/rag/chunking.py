"""Split markdown policy documents into heading-aware, overlapping chunks.

Each chunk keeps the document title and section heading so that:
  * retrieval has extra context ("Late Payment" in a loan doc vs a card doc), and
  * every answer can cite an exact source: <doc_id>#<section-slug>.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Chunk:
    chunk_id: str      # e.g. credit-card-terms#late-payment-charge:0
    doc_id: str        # file stem, e.g. credit-card-terms
    doc_title: str     # "Nusa Bank Credit Card Terms and Fees"
    section: str       # "Late Payment Charge"
    text: str

    @property
    def citation(self) -> str:
        return f"{self.doc_id}#{slugify(self.section)}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation"] = self.citation
        return d


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _split_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    title = "Untitled"
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Overview"
    buffer: list[str] = []

    for line in markdown.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
            continue
        if line.startswith("> "):  # disclaimer quote blocks are not policy content
            continue
        if line.startswith("## "):
            if buffer:
                sections.append((current_heading, buffer))
            current_heading, buffer = line[3:].strip(), []
            continue
        buffer.append(line)
    if buffer:
        sections.append((current_heading, buffer))

    cleaned = [(h, "\n".join(b).strip()) for h, b in sections]
    return title, [(h, body) for h, body in cleaned if body]


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        out.append(words[start : start + size])
        if start + size >= len(words):
            break
    return out


def chunk_document(path: Path, size: int = 120, overlap: int = 30) -> list[Chunk]:
    doc_id = path.stem
    title, sections = _split_sections(path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for heading, body in sections:
        for i, window in enumerate(_windows(body.split(), size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{slugify(heading)}:{i}",
                    doc_id=doc_id,
                    doc_title=title,
                    section=heading,
                    text=" ".join(window),
                )
            )
    return chunks


def chunk_directory(directory: Path, size: int = 120, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_document(path, size, overlap))
    return chunks
