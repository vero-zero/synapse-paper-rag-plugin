"""논문을 RAG용 청크로 만드는 로직.

- 문장 경계를 지키며 chunk_size 글자 안쪽으로 자르고, 앞 청크 끝부분을 overlap만큼 겹친다.
  (겹치지 않으면 경계에 걸친 문맥이 검색에서 사라진다)
- DOI, 없으면 정규화한 제목으로 중복 논문을 거른다.
- 걸러진 건수와 이유를 QualityReport로 남겨, 데이터셋을 받는 사람이 품질을 판단할 수 있게 한다.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel, Field

from plugin.core.openalex import Paper

_SENTENCE_END = re.compile(r"(?<=[.!?。])\s+")
_SPACES = re.compile(r"\s+")
_NON_WORD = re.compile(r"[\W_]+")


class Chunk(BaseModel):
    """chunks.jsonl의 한 줄. text는 임베딩 대상, metadata는 검색 필터·출처 표시용."""

    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    chunk_count: int
    metadata: dict


class QualityReport(BaseModel):
    input_papers: int = 0
    output_papers: int = 0
    output_chunks: int = 0
    skipped: Counter = Field(default_factory=Counter)
    languages: Counter = Field(default_factory=Counter)
    avg_chunk_chars: float = 0.0


def normalize(text: str) -> str:
    return _SPACES.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(normalize(text)) if s]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """문장 단위로 모아 chunk_size 이하의 청크를 만든다. 한 문장이 chunk_size보다 길면 글자 단위로 자른다."""
    if overlap >= chunk_size:
        raise ValueError("overlap은 chunk_size보다 작아야 합니다.")

    pieces: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) <= chunk_size:
            pieces.append(sentence)
        else:
            step = chunk_size - overlap
            pieces.extend(sentence[i : i + chunk_size] for i in range(0, len(sentence), step))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current} {piece}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # 겹치는 부분이 단어 중간에서 시작하지 않도록 첫 공백 뒤부터 사용
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            current = f"{tail} {piece}".strip() if len(tail) + len(piece) + 1 <= chunk_size else piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def dedupe_key(paper: Paper) -> str:
    """DOI가 있으면 DOI, 없으면 구두점·공백을 뺀 제목. 제목이 비면 OpenAlex ID로 구분한다."""
    if paper.doi:
        return f"doi:{paper.doi.lower()}"
    # 모든 언어의 글자를 남기고(\w는 유니코드 기준) 구두점·공백·밑줄만 제거한다.
    # NFKC는 전각/반각 같은 표기 차이를 맞추고, casefold는 lower보다 넓게 대소문자를 맞춘다.
    title = _NON_WORD.sub("", unicodedata.normalize("NFKC", paper.title).casefold())
    if not title:
        return f"id:{paper.source_id}"
    return f"title:{title}"


def doc_id_of(paper: Paper) -> str:
    return hashlib.sha1(dedupe_key(paper).encode()).hexdigest()[:16]


def build_chunks(
    papers: Iterable[Paper],
    *,
    chunk_size: int = 800,
    overlap: int = 100,
    min_abstract_chars: int = 200,
) -> tuple[list[Chunk], QualityReport]:
    report = QualityReport()
    seen: set[str] = set()
    chunks: list[Chunk] = []

    for paper in papers:
        report.input_papers += 1
        key = dedupe_key(paper)
        if key in seen:
            report.skipped["duplicate"] += 1
            continue
        seen.add(key)
        if not paper.abstract or len(paper.abstract) < min_abstract_chars:
            report.skipped["short_or_missing_abstract"] += 1
            continue

        # 제목을 본문 앞에 붙여야 청크 하나만 검색돼도 어떤 논문인지 알 수 있다
        texts = chunk_text(f"{paper.title}. {paper.abstract}", chunk_size, overlap)
        doc_id = doc_id_of(paper)
        metadata = paper.model_dump(exclude={"abstract"})
        for i, text in enumerate(texts):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}-{i:03d}",
                    doc_id=doc_id,
                    text=text,
                    chunk_index=i,
                    chunk_count=len(texts),
                    metadata=metadata,
                )
            )
        report.output_papers += 1
        report.languages[paper.language or "unknown"] += 1

    report.output_chunks = len(chunks)
    if chunks:
        report.avg_chunk_chars = round(sum(len(c.text) for c in chunks) / len(chunks), 1)
    return chunks, report
