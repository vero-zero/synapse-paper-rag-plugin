"""OpenAlex 논문 검색 클라이언트.

- 외부 응답을 내부 모델(Paper)로 정규화해서, 외부 규격이 바뀌어도 영향이 이 파일 안에 머물게 한다.
- 429/5xx/네트워크 오류는 지수 백오프로 재시도하고, 끝내 실패하면 SourceUnavailable을 던진다.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import httpx
from pydantic import BaseModel

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRY_AFTER_SECONDS = 60.0  # 서버가 너무 긴 대기를 요구해도 작업이 멈춰 있지 않도록 상한을 둔다


def _retry_after(error: Exception | None) -> float | None:
    """429/503 응답의 Retry-After(초 단위)를 읽는다. 없거나 날짜 형식이면 None을 돌려 지수 백오프를 쓴다."""
    if not isinstance(error, httpx.HTTPStatusError):
        return None
    value = error.response.headers.get("Retry-After")
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None
    if seconds is None or seconds < 0:
        return None
    return min(seconds, MAX_RETRY_AFTER_SECONDS)


SELECT_FIELDS = ",".join(
    [
        "id",
        "doi",
        "title",
        "publication_year",
        "language",
        "authorships",
        "primary_location",
        "open_access",
        "abstract_inverted_index",
        "cited_by_count",
    ]
)
MAX_PER_PAGE = 200  # OpenAlex 한 페이지 최대 건수


class SourceUnavailable(Exception):
    """외부 검색 소스를 일시적으로 사용할 수 없음."""


class Paper(BaseModel):
    """수집한 논문 한 건 (papers.jsonl의 한 줄)."""

    source_id: str
    title: str
    abstract: str | None
    doi: str | None = None
    year: int | None = None
    language: str | None = None
    authors: list[str] = []
    venue: str | None = None
    license: str | None = None
    oa_url: str | None = None
    cited_by_count: int = 0


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex는 저작권 때문에 초록을 {단어: [위치...]} 역색인으로 준다. 원래 문장으로 되돌린다."""
    if not inverted_index:
        return None
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions)) or None


def to_paper(work: dict[str, Any]) -> Paper:
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    open_access = work.get("open_access") or {}
    doi = work.get("doi")
    return Paper(
        source_id=(work.get("id") or "").rsplit("/", 1)[-1],
        title=(work.get("title") or "").strip() or "(제목 없음)",
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        doi=doi.removeprefix("https://doi.org/") if doi else None,
        year=work.get("publication_year"),
        language=work.get("language"),
        authors=[
            a["author"]["display_name"]
            for a in work.get("authorships") or []
            if (a.get("author") or {}).get("display_name")
        ][:20],
        venue=source.get("display_name"),
        license=location.get("license"),
        oa_url=open_access.get("oa_url"),
        cited_by_count=work.get("cited_by_count") or 0,
    )


class OpenAlexClient:
    def __init__(
        self,
        client: httpx.Client,
        base_url: str = "https://api.openalex.org",
        api_key: str | None = None,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        page_size: int = MAX_PER_PAGE,
    ):
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._page_size = min(page_size, MAX_PER_PAGE)

    def iter_papers(
        self,
        query: str,
        *,
        max_papers: int,
        language: str | None = None,
        from_year: int | None = None,
    ) -> Iterator[tuple[Paper, int]]:
        """조건에 맞는 오픈액세스 논문을 최대 max_papers건까지 (논문, 전체 검색 건수)로 돌려준다."""
        filters = ["is_oa:true", "has_abstract:true"]
        if language:
            filters.append(f"language:{language}")
        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")

        per_page = min(self._page_size, max_papers)
        page, yielded = 1, 0
        while yielded < max_papers:
            params: dict[str, Any] = {
                "search": query,
                "filter": ",".join(filters),
                "select": SELECT_FIELDS,
                "per-page": per_page,
                "page": page,
            }
            if self._api_key:
                params["api_key"] = self._api_key
            data = self._get("/works", params)
            results = data.get("results") or []
            total = (data.get("meta") or {}).get("count", 0)
            for work in results:
                yield to_paper(work), total
                yielded += 1
                if yielded >= max_papers:
                    return
            if len(results) < per_page:  # 마지막 페이지
                return
            page += 1

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.get(url, params=params)
                if resp.status_code not in RETRYABLE_STATUS:
                    resp.raise_for_status()
                    return resp.json()
                last_error = httpx.HTTPStatusError(
                    f"retryable status {resp.status_code}", request=resp.request, response=resp
                )
            except httpx.TransportError as e:
                last_error = e
            if attempt < self._max_retries:
                wait = _retry_after(last_error)
                if wait is None:
                    wait = self._backoff * (2**attempt)
                log.warning("OpenAlex 호출 실패(%s), %.1fs 후 재시도 %d/%d", last_error, wait, attempt + 1, self._max_retries)
                time.sleep(wait)
        raise SourceUnavailable(f"OpenAlex에 연결할 수 없습니다: {last_error}") from last_error
