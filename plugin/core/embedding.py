"""텍스트 임베딩 백엔드.

- hash: 외부 의존성 없이 단어를 해시해 고정 차원 벡터로 만든다. 결과가 항상 같아
  테스트·CI·오프라인 데모의 기준선으로 쓴다. 의미는 모르고 단어 겹침만 본다.
- openai_compatible: /embeddings 규격(OpenAI, Ollama, vLLM, TEI 등)을 HTTP로 호출한다.

모든 백엔드는 L2 정규화된 float32 행렬을 돌려주므로, 검색 쪽은 내적만으로 코사인 유사도를 얻는다.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import unicodedata
from collections.abc import Sequence
from typing import Literal, Protocol

import httpx
import numpy as np

from plugin.core.openalex import RETRYABLE_STATUS, _retry_after

log = logging.getLogger(__name__)

Provider = Literal["hash", "openai_compatible"]
_WORD = re.compile(r"\w+")


class EmbeddingError(Exception):
    """임베딩 API 호출이 실패했거나 응답이 규격과 다름."""


class Embedder(Protocol):
    provider: str
    model: str
    dimensions: int | None

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1.0, norms)).astype(np.float32)


class HashEmbedder:
    provider = "hash"
    model = "hash-bow-v1"

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _WORD.findall(unicodedata.normalize("NFKC", text).casefold()):
                h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
                # 부호도 해시로 정해 서로 다른 단어가 같은 칸에 부딪혀도 평균적으로 상쇄되게 한다
                out[row, h % self.dimensions] += 1.0 if (h >> 63) & 1 else -1.0
        return l2_normalize(out)


class OpenAICompatibleEmbedder:
    provider = "openai_compatible"

    def __init__(
        self,
        client: httpx.Client,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 64,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
    ):
        self._client = client
        self._url = f"{base_url.rstrip('/')}/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.model = model
        self.dimensions = dimensions
        self._requested_dimensions = dimensions
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._backoff = backoff_seconds

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        batches = [self._embed_batch(list(texts[i : i + self._batch_size])) for i in range(0, len(texts), self._batch_size)]
        if not batches:
            return np.zeros((0, self.dimensions or 0), dtype=np.float32)
        return l2_normalize(np.vstack(batches))

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        payload: dict = {"model": self.model, "input": batch}
        if self._requested_dimensions:
            payload["dimensions"] = self._requested_dimensions
        data = self._post(payload).get("data") or []
        if len(data) != len(batch):
            raise EmbeddingError(f"응답 개수가 입력과 다릅니다: 입력 {len(batch)}건, 응답 {len(data)}건")

        vectors = np.asarray([item["embedding"] for item in sorted(data, key=lambda d: d["index"])], dtype=np.float32)
        dim = vectors.shape[1]
        if self.dimensions is None:
            self.dimensions = dim  # 첫 응답에서 모델 차원을 확정하고, 이후 배치는 같은 차원이어야 한다
        elif dim != self.dimensions:
            raise EmbeddingError(f"임베딩 차원이 다릅니다: 기대 {self.dimensions}, 응답 {dim}")
        return vectors

    def _post(self, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post(self._url, json=payload, headers=self._headers)
                if resp.status_code not in RETRYABLE_STATUS:
                    if resp.is_error:
                        # 4xx는 재시도해도 같으므로 바로 실패시킨다. 본문은 원인 파악용으로 앞부분만 남긴다
                        raise EmbeddingError(f"임베딩 API 오류 {resp.status_code}: {resp.text[:200]}")
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
                log.warning("임베딩 호출 실패(%s), %.1fs 후 재시도 %d/%d", last_error, wait, attempt + 1, self._max_retries)
                time.sleep(wait)
        raise EmbeddingError(f"임베딩 API에 연결할 수 없습니다: {last_error}") from last_error


def get_embedder(
    provider: Provider,
    *,
    client: httpx.Client | None = None,
    model: str | None = None,
    dimensions: int | None = None,
    base_url: str = "https://api.openai.com/v1",
    api_key: str | None = None,
    batch_size: int = 64,
) -> Embedder:
    if provider == "hash":
        return HashEmbedder(dimensions=dimensions or 256)
    if client is None or not model:
        raise ValueError("openai_compatible 임베더에는 HTTP 클라이언트와 model이 필요합니다.")
    return OpenAICompatibleEmbedder(
        client, base_url=base_url, model=model, api_key=api_key, dimensions=dimensions, batch_size=batch_size
    )
