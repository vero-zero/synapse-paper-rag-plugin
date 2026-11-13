"""embed 액션: chunks.jsonl의 text를 임베딩해 embeddings.npy로 저장한다."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel, Field, model_validator

from synapse_sdk.plugins.action import BaseAction
from synapse_sdk.plugins.enums import PluginCategory

from plugin.collect import DEFAULT_STORAGE
from plugin.core.embedding import Provider, get_embedder
from plugin.core.io import TempDir, read_jsonl_with_hash, upload_files

API_KEY_ENV = "EMBEDDING_API_KEY"


class EmbedParams(BaseModel):
    dataset_name: str = Field(default="paper-rag", pattern=r"^[\w\-]+$", description="build_rag에서 쓴 폴더 이름")
    provider: Provider = Field(default="hash", description="hash(오프라인 기준선) 또는 openai_compatible")
    model: str | None = Field(default=None, description="openai_compatible일 때 모델 이름")
    dimensions: int | None = Field(default=None, ge=8, le=8192, description="벡터 차원 (hash 기본 256)")
    batch_size: int = Field(default=64, ge=1, le=2048, description="한 번에 임베딩할 청크 수")
    base_url: str = Field(default="https://api.openai.com/v1", description="openai_compatible API 주소")
    api_key: str | None = Field(default=None, description=f"없으면 {API_KEY_ENV} 환경변수를 쓴다")
    storage: dict[str, Any] = Field(default_factory=lambda: dict(DEFAULT_STORAGE))

    @model_validator(mode="after")
    def check_model(self) -> EmbedParams:
        if self.provider == "openai_compatible" and not self.model:
            raise ValueError("provider가 openai_compatible이면 model을 지정해야 합니다.")
        return self


class EmbedResult(BaseModel):
    dataset_name: str
    provider: str
    model: str
    count: int
    dimensions: int
    embeddings_path: str
    manifest_path: str


class EmbedAction(BaseAction[EmbedParams]):
    action_name = "embed"
    category = PluginCategory.CUSTOM
    result_model = EmbedResult

    def execute(self) -> EmbedResult:
        p = self.params
        chunks, chunks_sha = read_jsonl_with_hash(p.storage, f"{p.dataset_name}/chunks.jsonl")
        if not chunks:
            raise ValueError("chunks.jsonl에 청크가 없습니다. build_rag 결과를 확인하세요.")
        texts = [c["text"] for c in chunks]

        with httpx.Client(timeout=60) as http, TempDir() as tmp:
            embedder = get_embedder(
                p.provider,
                client=http,
                model=p.model,
                dimensions=p.dimensions,
                base_url=p.base_url,
                api_key=p.api_key or os.environ.get(API_KEY_ENV),
                batch_size=p.batch_size,
            )
            parts = []
            for start in range(0, len(texts), p.batch_size):
                parts.append(embedder.embed(texts[start : start + p.batch_size]))
                self.set_progress(min(start + p.batch_size, len(texts)), len(texts), step="embed")
            matrix = np.vstack(parts)

            manifest = {
                "dataset_name": p.dataset_name,
                "created_at": datetime.now(UTC).isoformat(),
                "provider": embedder.provider,
                "model": embedder.model,
                "dimensions": int(matrix.shape[1]),
                "count": int(matrix.shape[0]),
                "dtype": "float32",
                "normalized": True,
                "chunks_sha256": chunks_sha,
                # api_key·storage는 비밀정보·환경 설정이라 남기지 않는다
                "params": p.model_dump(exclude={"api_key", "storage"}),
            }
            files = {
                "embeddings.npy": tmp / "embeddings.npy",
                "embedding_ids.json": tmp / "embedding_ids.json",
                "embeddings_manifest.json": tmp / "embeddings_manifest.json",
            }
            np.save(files["embeddings.npy"], matrix)
            ids = {"chunk_ids": [c["chunk_id"] for c in chunks], "doc_ids": [c["doc_id"] for c in chunks]}
            files["embedding_ids.json"].write_text(json.dumps(ids), encoding="utf-8")
            files["embeddings_manifest.json"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            saved = upload_files(p.storage, p.dataset_name, files)

        self.set_metrics({"count": manifest["count"], "dimensions": manifest["dimensions"]}, step="embed")
        self.log("embed_done", {"provider": manifest["provider"], "model": manifest["model"], "count": manifest["count"]})
        return EmbedResult(
            dataset_name=p.dataset_name,
            provider=manifest["provider"],
            model=manifest["model"],
            count=manifest["count"],
            dimensions=manifest["dimensions"],
            embeddings_path=saved["embeddings.npy"],
            manifest_path=saved["embeddings_manifest.json"],
        )
