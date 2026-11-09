"""collect 액션: OpenAlex에서 논문을 수집해 papers.jsonl로 저장한다."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field

from synapse_sdk.plugins.action import BaseAction
from synapse_sdk.plugins.enums import PluginCategory

from plugin.core.io import TempDir, upload_files, write_jsonl
from plugin.core.openalex import OpenAlexClient

DEFAULT_STORAGE = {"provider": "local", "configuration": {"location": "./output"}}


class CollectParams(BaseModel):
    query: str = Field(min_length=2, description="검색어 (제목·초록·본문)")
    max_papers: int = Field(default=50, ge=1, le=1000, description="최대 수집 건수")
    language: str | None = Field(default=None, description="언어 코드 필터 (예: en, ko)")
    from_year: int | None = Field(default=None, ge=1900, le=2100, description="이 연도 이후 출판")
    dataset_name: str = Field(default="paper-rag", pattern=r"^[\w\-]+$", description="저장 폴더 이름")
    storage: dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_STORAGE), description="Synapse Storage 설정")
    openalex_base_url: str = "https://api.openalex.org"
    openalex_api_key: str | None = None


class CollectResult(BaseModel):
    dataset_name: str
    collected: int
    total_matched: int
    papers_path: str


class CollectAction(BaseAction[CollectParams]):
    action_name = "collect"
    category = PluginCategory.CUSTOM
    result_model = CollectResult

    def execute(self) -> CollectResult:
        p = self.params
        self.log("collect_start", {"query": p.query, "max_papers": p.max_papers, "language": p.language})

        with httpx.Client(timeout=30) as http, TempDir() as tmp:
            client = OpenAlexClient(http, base_url=p.openalex_base_url, api_key=p.openalex_api_key)
            papers, total = [], 0
            for paper, total in client.iter_papers(
                p.query, max_papers=p.max_papers, language=p.language, from_year=p.from_year
            ):
                papers.append(paper)
                self.set_progress(len(papers), p.max_papers, step="collect")

            local = tmp / "papers.jsonl"
            write_jsonl(local, papers)
            saved = upload_files(p.storage, p.dataset_name, {"papers.jsonl": local})

        self.set_metrics({"collected": len(papers), "total_matched": total}, step="collect")
        self.log("collect_done", {"collected": len(papers), "path": saved["papers.jsonl"]})
        return CollectResult(
            dataset_name=p.dataset_name,
            collected=len(papers),
            total_matched=total,
            papers_path=saved["papers.jsonl"],
        )
