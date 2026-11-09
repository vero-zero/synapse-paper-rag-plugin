"""build_rag 액션: papers.jsonl을 RAG용 chunks.jsonl과 manifest.json으로 만든다."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from synapse_sdk.plugins.action import BaseAction
from synapse_sdk.plugins.enums import PluginCategory

from plugin.collect import DEFAULT_STORAGE
from plugin.core.chunking import build_chunks
from plugin.core.io import TempDir, iter_jsonl, read_text_lines, upload_files, write_jsonl
from plugin.core.openalex import Paper


class BuildRagParams(BaseModel):
    dataset_name: str = Field(default="paper-rag", pattern=r"^[\w\-]+$", description="collect에서 쓴 폴더 이름")
    chunk_size: int = Field(default=800, ge=100, le=8000, description="청크 최대 글자 수")
    chunk_overlap: int = Field(default=100, ge=0, description="앞 청크와 겹칠 글자 수")
    min_abstract_chars: int = Field(default=200, ge=0, description="이보다 짧은 초록은 제외")
    storage: dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_STORAGE))

    @model_validator(mode="after")
    def check_overlap(self) -> BuildRagParams:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap은 chunk_size보다 작아야 합니다.")
        return self


class BuildRagResult(BaseModel):
    dataset_name: str
    papers: int
    chunks: int
    skipped: dict[str, int]
    chunks_path: str
    manifest_path: str


class BuildRagAction(BaseAction[BuildRagParams]):
    action_name = "build_rag"
    category = PluginCategory.CUSTOM
    result_model = BuildRagResult

    def execute(self) -> BuildRagResult:
        p = self.params
        lines = read_text_lines(p.storage, f"{p.dataset_name}/papers.jsonl")
        papers = [Paper.model_validate(row) for row in iter_jsonl(lines)]
        self.set_progress(1, 3, step="load")

        chunks, report = build_chunks(
            papers, chunk_size=p.chunk_size, overlap=p.chunk_overlap, min_abstract_chars=p.min_abstract_chars
        )
        self.set_progress(2, 3, step="chunk")

        manifest = {
            "dataset_name": p.dataset_name,
            "created_at": datetime.now(UTC).isoformat(),
            "format": "jsonl",
            "schema": {"text": "임베딩 대상 텍스트", "metadata": "출처·저자·연도·DOI·라이선스"},
            "params": p.model_dump(exclude={"storage"}),
            "quality": {
                "input_papers": report.input_papers,
                "output_papers": report.output_papers,
                "output_chunks": report.output_chunks,
                "avg_chunk_chars": report.avg_chunk_chars,
                "skipped": dict(report.skipped),
                "languages": dict(report.languages),
            },
        }

        with TempDir() as tmp:
            chunks_file, manifest_file = tmp / "chunks.jsonl", tmp / "manifest.json"
            write_jsonl(chunks_file, chunks)
            manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            saved = upload_files(p.storage, p.dataset_name, {"chunks.jsonl": chunks_file, "manifest.json": manifest_file})
        self.set_progress(3, 3, step="save")

        self.set_metrics(manifest["quality"] | {"skipped_total": sum(report.skipped.values())}, step="build_rag")
        self.log("build_rag_done", {"chunks": report.output_chunks, "skipped": dict(report.skipped)})
        return BuildRagResult(
            dataset_name=p.dataset_name,
            papers=report.output_papers,
            chunks=report.output_chunks,
            skipped=dict(report.skipped),
            chunks_path=saved["chunks.jsonl"],
            manifest_path=saved["manifest.json"],
        )
