"""Synapse SDK의 LocalExecutor로 액션을 실제 플러그인처럼 실행하는 통합 테스트."""

import json

import httpx
import pytest
import respx

from synapse_sdk.plugins.discovery import PluginDiscovery
from synapse_sdk.plugins.errors import ExecutionError
from synapse_sdk.plugins.executors.local import LocalExecutor

from plugin.build_rag import BuildRagAction
from plugin.collect import CollectAction
from plugin.embed import EmbedAction
from plugin.evaluate import EvaluateAction
from tests.test_openalex import BASE, work


@pytest.fixture
def storage(tmp_path):
    return {"provider": "local", "configuration": {"location": str(tmp_path)}}


def long_work(i: int) -> dict:
    w = work(i)
    w["abstract_inverted_index"] = {f"word{j}": [j] for j in range(120)}  # 긴 초록
    return w


def test_config_yaml_declares_all_actions():
    discovery = PluginDiscovery.from_path(".")
    assert discovery.list_actions() == ["collect", "build_rag", "embed", "evaluate"]
    assert discovery.get_action_class("collect") is CollectAction
    assert discovery.get_action_class("build_rag") is BuildRagAction
    assert discovery.get_action_class("embed") is EmbedAction
    assert discovery.get_action_class("evaluate") is EvaluateAction


@respx.mock
def test_collect_then_build_rag_end_to_end(tmp_path, storage):
    respx.get(f"{BASE}/works").mock(
        return_value=httpx.Response(200, json={"meta": {"count": 3}, "results": [long_work(1), long_work(1), long_work(2)]})
    )
    executor = LocalExecutor()

    collected = executor.execute(
        CollectAction, {"query": "rag", "max_papers": 3, "dataset_name": "t1", "storage": storage, "openalex_base_url": BASE}
    )
    assert collected.collected == 3
    assert (tmp_path / "t1" / "papers.jsonl").exists()

    built = executor.execute(
        BuildRagAction, {"dataset_name": "t1", "chunk_size": 300, "chunk_overlap": 50, "storage": storage}
    )
    assert built.papers == 2
    assert built.skipped == {"duplicate": 1}

    lines = (tmp_path / "t1" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == built.chunks
    assert json.loads(lines[0])["metadata"]["license"] == "cc-by"

    manifest = json.loads((tmp_path / "t1" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["quality"]["output_chunks"] == built.chunks
    assert manifest["params"]["chunk_size"] == 300


def test_build_rag_rejects_overlap_not_smaller_than_chunk_size(storage):
    # SDK가 파라미터 검증 실패를 ExecutionError로 감싸서 올린다
    with pytest.raises(ExecutionError, match="chunk_overlap"):
        LocalExecutor().execute(BuildRagAction, {"chunk_size": 200, "chunk_overlap": 200, "storage": storage})


def test_collect_rejects_unsafe_dataset_name(storage):
    # 경로 조작(../)으로 저장 위치를 벗어나지 못하게 막는다
    with pytest.raises(ExecutionError, match="dataset_name"):
        LocalExecutor().execute(CollectAction, {"query": "rag", "dataset_name": "../etc", "storage": storage})
