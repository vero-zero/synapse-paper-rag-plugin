"""벡터 검색과 검색 품질 지표.

- 검색은 정규화된 벡터의 내적(=코사인 유사도)을 전부 계산하는 brute-force 방식이다.
  수만 청크까지는 numpy 행렬곱으로 충분하고, 그 이상은 벡터 DB의 몫이다.
- 정답 판정은 청크가 아니라 문서(doc_id) 단위로 한다. 한 논문의 청크가 상위권을
  여러 칸 차지해도 사용자 입장에서는 같은 논문 하나를 찾은 것이기 때문이다.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from plugin.core.chunking import normalize, split_sentences

AutoQueryMode = Literal["title", "first_sentence"]


class EvalQuery(BaseModel):
    """평가 질의 한 건. eval_queries.jsonl의 한 줄 형식과 같다."""

    query: str = Field(min_length=1)
    relevant_doc_ids: list[str] = Field(min_length=1)


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(relevant.intersection(ranked[:k])) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(ranked):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """이진 관련도 nDCG. 순위가 낮을수록 log2로 할인한다."""
    dcg = sum(1.0 / math.log2(i + 2) for i, doc_id in enumerate(ranked[:k]) if doc_id in relevant)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


def search(query_vecs: np.ndarray, matrix: np.ndarray, doc_ids: Sequence[str], top_k: int) -> list[list[str]]:
    """질의마다 유사도가 높은 문서 doc_id를 top_k개까지 돌려준다. 문서 점수는 그 문서 청크 중 최고점이다."""
    unique_docs, doc_index = np.unique(np.asarray(doc_ids), return_inverse=True)
    scores = query_vecs @ matrix.T  # (질의 수, 청크 수)
    doc_scores = np.full((len(query_vecs), len(unique_docs)), -np.inf, dtype=np.float32)
    np.maximum.at(doc_scores.T, doc_index, scores.T)
    # 동점이면 doc_id 순서로 정해지도록 stable 정렬
    order = np.argsort(-doc_scores, axis=1, kind="stable")[:, :top_k]
    return [[str(unique_docs[j]) for j in row] for row in order]


def evaluate_rankings(
    queries: Sequence[EvalQuery], rankings: Sequence[Sequence[str]], k_values: Sequence[int], max_misses: int = 5
) -> dict[str, Any]:
    """질의별 지표를 평균 내고, 상위 max(k) 안에서 정답을 못 찾은 질의 몇 개를 디버깅용으로 남긴다."""
    max_k = max(k_values)
    totals: dict[str, float] = {f"recall@{k}": 0.0 for k in k_values}
    totals |= {f"ndcg@{k}": 0.0 for k in k_values}
    totals["mrr"] = 0.0
    misses: list[dict[str, Any]] = []

    for q, ranked in zip(queries, rankings, strict=True):
        relevant = set(q.relevant_doc_ids)
        for k in k_values:
            totals[f"recall@{k}"] += recall_at_k(ranked, relevant, k)
            totals[f"ndcg@{k}"] += ndcg_at_k(ranked, relevant, k)
        rr = reciprocal_rank(ranked[:max_k], relevant)
        totals["mrr"] += rr
        if rr == 0.0 and len(misses) < max_misses:
            misses.append({"query": q.query, "relevant_doc_ids": q.relevant_doc_ids, "top_doc_ids": list(ranked[:3])})

    n = len(queries)
    metrics = {name: round(value / n, 4) if n else 0.0 for name, value in totals.items()}
    return {"metrics": metrics, "misses": misses}


def auto_queries(
    chunks: Iterable[dict[str, Any]], *, mode: AutoQueryMode, max_queries: int, seed: int = 0
) -> tuple[list[EvalQuery], int]:
    """정답셋이 없을 때 문서마다 질의 하나를 만든다 (known-item 검색). (질의 목록, 질의를 못 만든 문서 수)를 돌려준다.

    - title: 논문 제목을 질의로 쓴다.
    - first_sentence: 초록 첫 문장을 질의로 쓴다. 첫 문장이 chunk_size보다 길어 글자 단위로
      잘린 문서는 0번 청크에 제목만 남으므로 건너뛰고 건수만 센다.
    두 방식 모두 질의 문장이 청크 본문에 그대로 들어 있어 점수가 실제보다 높게 나온다.
    임베딩 설정을 서로 비교하는 기준선 용도이고, 실제 품질은 사람이 만든 정답셋으로 봐야 한다.
    """
    candidates: list[EvalQuery] = []
    skipped = 0
    for chunk in chunks:
        if chunk.get("chunk_index") != 0:
            continue
        title = normalize(chunk.get("metadata", {}).get("title") or "")
        query = title if mode == "title" else _first_sentence(chunk["text"], title)
        if not query:
            skipped += 1
            continue
        candidates.append(EvalQuery(query=query, relevant_doc_ids=[chunk["doc_id"]]))

    if len(candidates) > max_queries:
        picked = sorted(random.Random(seed).sample(range(len(candidates)), max_queries))
        candidates = [candidates[i] for i in picked]
    return candidates, skipped


def _first_sentence(text: str, title: str) -> str | None:
    """build_chunks가 붙인 "제목. " 접두어를 떼고 초록 첫 문장을 돌려준다."""
    prefix = f"{title}."
    if not title or not text.startswith(prefix):
        return None
    sentences = split_sentences(text[len(prefix) :].lstrip(". "))
    return sentences[0] if sentences else None
