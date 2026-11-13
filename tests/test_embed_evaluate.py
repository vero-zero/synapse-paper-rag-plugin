"""embed → evaluate 액션을 LocalExecutor로 실제 플러그인처럼 실행하는 통합 테스트."""

import json

import httpx
import numpy as np
import pytest
import respx

from synapse_sdk.plugins.errors import ExecutionError
from synapse_sdk.plugins.executors.local import LocalExecutor

from plugin.build_rag import BuildRagAction
from plugin.embed import EmbedAction
from plugin.evaluate import EvaluateAction

TOPICS = {
    "W1": ("Dense passage retrieval", "dense vectors encode questions and passages for open domain answering"),
    "W2": ("Protein structure prediction", "amino acid sequences fold into three dimensional protein shapes"),
    "W3": ("Coral reef bleaching", "ocean warming stresses coral symbionts and reef ecosystems decline"),
    "W4": ("Battery cathode materials", "lithium nickel oxide cathodes improve energy density of cells"),
}


@pytest.fixture
def storage(tmp_path):
    return {"provider": "local", "configuration": {"location": str(tmp_path)}}


@pytest.fixture
def built(tmp_path, storage):
    """서로 다른 주제의 논문 4편으로 chunks.jsonl까지 만들어 둔다."""
    folder = tmp_path / "ds"
    folder.mkdir()
    with (folder / "papers.jsonl").open("w", encoding="utf-8") as f:
        for source_id, (title, words) in TOPICS.items():
            abstract = " ".join(f"{words.capitalize()} in study {n}." for n in range(4))
            paper = {"source_id": source_id, "title": title, "abstract": abstract, "doi": f"10.1/{source_id}"}
            f.write(json.dumps(paper) + "\n")
    LocalExecutor().execute(
        BuildRagAction, {"dataset_name": "ds", "chunk_size": 200, "chunk_overlap": 20, "min_abstract_chars": 10, "storage": storage}
    )
    return folder


def test_embed_then_evaluate_with_hash_provider(built, storage):
    executor = LocalExecutor()
    embedded = executor.execute(EmbedAction, {"dataset_name": "ds", "dimensions": 128, "storage": storage})

    chunk_count = len((built / "chunks.jsonl").read_text(encoding="utf-8").splitlines())
    assert embedded.count == chunk_count
    assert embedded.dimensions == 128
    matrix = np.load(built / "embeddings.npy")
    assert matrix.shape == (chunk_count, 128)
    ids = json.loads((built / "embedding_ids.json").read_text(encoding="utf-8"))
    assert len(ids["chunk_ids"]) == len(ids["doc_ids"]) == chunk_count
    manifest = json.loads((built / "embeddings_manifest.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == "hash"
    assert len(manifest["chunks_sha256"]) == 64

    result = executor.execute(EvaluateAction, {"dataset_name": "ds", "k_values": [1, 3], "storage": storage})
    assert result.queries == 4
    assert result.query_source == "auto_first_sentence"
    assert result.metrics["recall@1"] == 1.0  # 주제가 완전히 달라 단어 겹침만으로도 찾아야 한다
    report = json.loads((built / "eval_report.json").read_text(encoding="utf-8"))
    assert report["metrics"] == result.metrics
    assert "warning" in report


def test_evaluate_uses_user_queries_file(built, storage):
    executor = LocalExecutor()
    executor.execute(EmbedAction, {"dataset_name": "ds", "storage": storage})
    doc_ids = json.loads((built / "embedding_ids.json").read_text(encoding="utf-8"))["doc_ids"]
    chunks = [json.loads(line) for line in (built / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    coral = next(c["doc_id"] for c in chunks if c["metadata"]["title"] == "Coral reef bleaching")
    rows = [
        {"query": "why do coral reefs bleach when the ocean warms", "relevant_doc_ids": [coral]},
        {"query": "unknown doc", "relevant_doc_ids": ["does-not-exist"]},
    ]
    (built / "gold.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    result = executor.execute(
        EvaluateAction, {"dataset_name": "ds", "queries_file": "gold.jsonl", "k_values": [1], "storage": storage}
    )
    assert coral in doc_ids
    assert result.query_source == "file"
    assert result.queries == 1
    assert result.skipped_queries == 1
    assert result.metrics["recall@1"] == 1.0
    assert "warning" not in json.loads((built / "eval_report.json").read_text(encoding="utf-8"))


def test_evaluate_rejects_embeddings_older_than_chunks(built, storage):
    executor = LocalExecutor()
    executor.execute(EmbedAction, {"dataset_name": "ds", "storage": storage})
    executor.execute(
        BuildRagAction, {"dataset_name": "ds", "chunk_size": 120, "chunk_overlap": 20, "min_abstract_chars": 10, "storage": storage}
    )
    with pytest.raises(ExecutionError, match="embed"):
        executor.execute(EvaluateAction, {"dataset_name": "ds", "storage": storage})


def test_embed_openai_compatible_requires_model(storage):
    with pytest.raises(ExecutionError, match="model"):
        LocalExecutor().execute(EmbedAction, {"dataset_name": "ds", "provider": "openai_compatible", "storage": storage})


def test_evaluate_rejects_path_in_queries_file(storage):
    with pytest.raises(ExecutionError, match="queries_file"):
        LocalExecutor().execute(EvaluateAction, {"dataset_name": "ds", "queries_file": "../x.jsonl", "storage": storage})


@respx.mock
def test_openai_compatible_end_to_end_keeps_key_out_of_files(built, storage):
    api = "https://embed.test/v1"
    vocab = sorted({w for _, words in TOPICS.values() for w in words.split()} | {t.lower() for t, _ in TOPICS.values()})

    def fake_embeddings(request):
        body = json.loads(request.content)
        assert body["model"] == "tiny-embed"
        # embed에서 차원을 지정하지 않았으면 evaluate 질의 요청에도 붙이면 안 된다 (지원하지 않는 모델이 거부함)
        assert "dimensions" not in body
        vecs = [[float(w in text.lower().split()) + 1e-3 for w in vocab] for text in body["input"]]
        return httpx.Response(200, json={"data": [{"index": i, "embedding": v} for i, v in enumerate(vecs)]})

    route = respx.post(f"{api}/embeddings").mock(side_effect=fake_embeddings)
    common = {"dataset_name": "ds", "base_url": api, "api_key": "sk-secret", "storage": storage}
    executor = LocalExecutor()
    embedded = executor.execute(EmbedAction, common | {"provider": "openai_compatible", "model": "tiny-embed", "batch_size": 2})
    assert embedded.dimensions == len(vocab)
    assert route.calls[0].request.headers["Authorization"] == "Bearer sk-secret"

    # evaluate는 provider·model을 매니페스트에서 읽으므로 따로 지정하지 않는다
    result = executor.execute(EvaluateAction, common | {"auto_query": "title", "k_values": [1]})
    assert result.queries == 4

    for name in ("embeddings_manifest.json", "eval_report.json"):
        assert "sk-secret" not in (built / name).read_text(encoding="utf-8")
