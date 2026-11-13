import httpx
import numpy as np
import pytest
import respx

from plugin.core.embedding import EmbeddingError, HashEmbedder, OpenAICompatibleEmbedder

API = "https://embed.test/v1"


def test_hash_embedder_is_deterministic_and_normalized():
    emb = HashEmbedder(dimensions=64)
    a = emb.embed(["dense retrieval works", "dense retrieval works", "完全に 다른 text"])
    assert a.shape == (3, 64)
    assert a.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(a[0], a[1])
    assert not np.array_equal(a[0], a[2])


def test_hash_embedder_similar_texts_score_higher():
    emb = HashEmbedder(dimensions=256)
    q, near, far = emb.embed(["dense passage retrieval", "dense passage retrieval for QA", "protein folding dynamics"])
    assert q @ near > q @ far


def test_hash_embedder_empty_text_gives_zero_vector():
    vec = HashEmbedder(dimensions=16).embed([""])
    assert not np.isnan(vec).any()
    assert np.linalg.norm(vec) == 0


def _data(vectors, order=None):
    order = order or list(range(len(vectors)))
    return {"data": [{"index": i, "embedding": vectors[i]} for i in order]}


@respx.mock
def test_openai_compatible_batches_and_restores_order():
    route = respx.post(f"{API}/embeddings").mock(
        side_effect=[
            httpx.Response(200, json=_data([[3.0, 4.0], [0.0, 2.0]], order=[1, 0])),
            httpx.Response(200, json=_data([[1.0, 0.0]])),
        ]
    )
    with httpx.Client() as http:
        emb = OpenAICompatibleEmbedder(http, base_url=API, model="m", api_key="secret", batch_size=2)
        vecs = emb.embed(["a", "b", "c"])

    assert route.call_count == 2
    assert route.calls[0].request.headers["Authorization"] == "Bearer secret"
    np.testing.assert_allclose(vecs, [[0.6, 0.8], [0.0, 1.0], [1.0, 0.0]], rtol=1e-6)
    assert emb.dimensions == 2


@respx.mock
def test_openai_compatible_retries_on_429(monkeypatch):
    monkeypatch.setattr("plugin.core.embedding.time.sleep", lambda s: None)
    respx.post(f"{API}/embeddings").mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=_data([[1.0, 0.0]]))]
    )
    with httpx.Client() as http:
        vecs = OpenAICompatibleEmbedder(http, base_url=API, model="m").embed(["a"])
    assert vecs.shape == (1, 2)


@respx.mock
def test_openai_compatible_gives_up_after_retries(monkeypatch):
    monkeypatch.setattr("plugin.core.embedding.time.sleep", lambda s: None)
    respx.post(f"{API}/embeddings").mock(return_value=httpx.Response(503))
    with httpx.Client() as http, pytest.raises(EmbeddingError, match="503"):
        OpenAICompatibleEmbedder(http, base_url=API, model="m", max_retries=1).embed(["a"])


@respx.mock
def test_openai_compatible_client_error_is_not_retried():
    route = respx.post(f"{API}/embeddings").mock(return_value=httpx.Response(401, json={"error": "bad key"}))
    with httpx.Client() as http, pytest.raises(EmbeddingError, match="401"):
        OpenAICompatibleEmbedder(http, base_url=API, model="m").embed(["a"])
    assert route.call_count == 1


@respx.mock
def test_openai_compatible_rejects_dimension_mismatch():
    respx.post(f"{API}/embeddings").mock(return_value=httpx.Response(200, json=_data([[1.0, 0.0, 0.0]])))
    with httpx.Client() as http, pytest.raises(EmbeddingError, match="차원"):
        OpenAICompatibleEmbedder(http, base_url=API, model="m", dimensions=2).embed(["a"])


@respx.mock
def test_openai_compatible_rejects_missing_items():
    respx.post(f"{API}/embeddings").mock(return_value=httpx.Response(200, json=_data([[1.0, 0.0]])))
    with httpx.Client() as http, pytest.raises(EmbeddingError, match="개수"):
        OpenAICompatibleEmbedder(http, base_url=API, model="m").embed(["a", "b"])
