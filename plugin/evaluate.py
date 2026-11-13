"""evaluate 액션: 저장된 임베딩으로 벡터 검색을 해 보고 Recall@k·MRR·nDCG@k를 eval_report.json으로 남긴다."""

from __future__ import annotations

import io
import json
import os
from datetime import UTC, datetime
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel, Field, field_validator

from synapse_sdk.plugins.action import BaseAction
from synapse_sdk.plugins.enums import PluginCategory

from plugin.collect import DEFAULT_STORAGE
from plugin.core.embedding import get_embedder
from plugin.core.io import TempDir, iter_jsonl, read_bytes, read_jsonl_with_hash, read_text_lines, upload_files
from plugin.core.retrieval import AutoQueryMode, EvalQuery, auto_queries, evaluate_rankings, search
from plugin.embed import API_KEY_ENV

AUTO_QUERY_WARNING = (
    "자동 생성 질의는 문장이 청크 본문에 그대로 들어 있어 점수가 실제보다 높게 나옵니다. "
    "임베딩 설정끼리 비교하는 기준선으로만 쓰고, 실제 품질은 queries_file 정답셋으로 확인하세요."
)


class EvaluateParams(BaseModel):
    dataset_name: str = Field(default="paper-rag", pattern=r"^[\w\-]+$", description="embed에서 쓴 폴더 이름")
    k_values: list[int] = Field(default_factory=lambda: [1, 5, 10], description="Recall@k·nDCG@k를 계산할 k 목록")
    queries_file: str | None = Field(
        default=None,
        pattern=r"^[\w\-.]+\.jsonl$",
        description="데이터셋 폴더 안의 정답셋 파일 (한 줄: {query, relevant_doc_ids}). 없으면 자동 생성",
    )
    auto_query: AutoQueryMode = Field(default="first_sentence", description="정답셋이 없을 때 질의 생성 방식")
    max_queries: int = Field(default=200, ge=1, le=10000, description="자동 생성 질의 최대 수")
    seed: int = Field(default=0, description="자동 질의 샘플링 시드")
    base_url: str = Field(default="https://api.openai.com/v1", description="openai_compatible 질의 임베딩 API 주소")
    api_key: str | None = Field(default=None, description=f"없으면 {API_KEY_ENV} 환경변수를 쓴다")
    storage: dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_STORAGE))

    @field_validator("k_values")
    @classmethod
    def check_k_values(cls, v: list[int]) -> list[int]:
        if not v or any(k < 1 or k > 100 for k in v):
            raise ValueError("k_values는 1~100 사이 정수를 하나 이상 담아야 합니다.")
        return sorted(set(v))


class EvaluateResult(BaseModel):
    dataset_name: str
    query_source: str
    queries: int
    skipped_queries: int
    metrics: dict[str, float]
    report_path: str


class EvaluateAction(BaseAction[EvaluateParams]):
    action_name = "evaluate"
    category = PluginCategory.CUSTOM
    result_model = EvaluateResult

    def execute(self) -> EvaluateResult:
        p = self.params
        prefix = p.dataset_name
        manifest = json.loads("".join(read_text_lines(p.storage, f"{prefix}/embeddings_manifest.json")))
        chunks, chunks_sha = read_jsonl_with_hash(p.storage, f"{prefix}/chunks.jsonl")
        if manifest["chunks_sha256"] != chunks_sha:
            raise ValueError("임베딩이 현재 chunks.jsonl과 맞지 않습니다. build_rag 이후 embed를 다시 실행하세요.")

        ids = json.loads("".join(read_text_lines(p.storage, f"{prefix}/embedding_ids.json")))
        matrix = np.load(io.BytesIO(read_bytes(p.storage, f"{prefix}/embeddings.npy")))
        if matrix.shape[0] != len(ids["doc_ids"]):
            raise ValueError(f"벡터 수({matrix.shape[0]})와 ID 수({len(ids['doc_ids'])})가 다릅니다.")
        self.set_progress(1, 3, step="load")

        queries, skipped, source = self._load_queries(p, chunks, set(ids["doc_ids"]))
        if not queries:
            raise ValueError("평가할 질의가 없습니다. queries_file의 doc_id가 데이터셋에 있는지 확인하세요.")

        # 질의는 반드시 청크와 같은 모델로 임베딩해야 하므로 provider·model을 매니페스트에서 가져온다.
        # 차원은 embed 때 요청한 값을 그대로 쓴다. 실제 차원을 요청에 붙이면 dimensions를 지원하지 않는 모델이 거부한다
        with httpx.Client(timeout=60) as http:
            embedder = get_embedder(
                manifest["provider"],
                client=http,
                model=manifest["model"],
                dimensions=manifest["params"].get("dimensions"),
                base_url=p.base_url,
                api_key=p.api_key or os.environ.get(API_KEY_ENV),
            )
            query_vecs = embedder.embed([q.query for q in queries])
        if query_vecs.shape[1] != matrix.shape[1]:
            raise ValueError(f"질의 벡터 차원({query_vecs.shape[1]})이 청크 벡터 차원({matrix.shape[1]})과 다릅니다.")
        self.set_progress(2, 3, step="embed_queries")

        rankings = search(query_vecs, matrix, ids["doc_ids"], top_k=max(p.k_values))
        evaluation = evaluate_rankings(queries, rankings, p.k_values)
        report: dict[str, Any] = {
            "dataset_name": prefix,
            "created_at": datetime.now(UTC).isoformat(),
            "embedding": {k: manifest[k] for k in ("provider", "model", "dimensions", "count")},
            "query_source": source,
            "queries": len(queries),
            "skipped_queries": skipped,
            "k_values": p.k_values,
            "metrics": evaluation["metrics"],
            "misses": evaluation["misses"],
        }
        if source.startswith("auto"):
            report["warning"] = AUTO_QUERY_WARNING

        with TempDir() as tmp:
            report_file = tmp / "eval_report.json"
            report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            saved = upload_files(p.storage, prefix, {"eval_report.json": report_file})
        self.set_progress(3, 3, step="save")

        self.set_metrics(evaluation["metrics"] | {"queries": len(queries)}, step="evaluate")
        self.log("evaluate_done", {"query_source": source, "queries": len(queries), "metrics": evaluation["metrics"]})
        return EvaluateResult(
            dataset_name=prefix,
            query_source=source,
            queries=len(queries),
            skipped_queries=skipped,
            metrics=evaluation["metrics"],
            report_path=saved["eval_report.json"],
        )

    @staticmethod
    def _load_queries(
        p: EvaluateParams, chunks: list[dict[str, Any]], known_docs: set[str]
    ) -> tuple[list[EvalQuery], int, str]:
        if p.queries_file is None:
            queries, skipped = auto_queries(chunks, mode=p.auto_query, max_queries=p.max_queries, seed=p.seed)
            return queries, skipped, f"auto_{p.auto_query}"

        queries, skipped = [], 0
        for row in iter_jsonl(read_text_lines(p.storage, f"{p.dataset_name}/{p.queries_file}")):
            q = EvalQuery.model_validate(row)
            relevant = [d for d in q.relevant_doc_ids if d in known_docs]
            if not relevant:
                skipped += 1  # 정답 문서가 데이터셋에 없으면 어떤 검색으로도 맞힐 수 없어 평균에서 뺀다
                continue
            queries.append(q.model_copy(update={"relevant_doc_ids": relevant}))
        return queries, skipped, "file"
