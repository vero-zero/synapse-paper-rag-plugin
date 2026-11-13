import math

import numpy as np
import pytest
from pydantic import ValidationError

from plugin.core.retrieval import (
    EvalQuery,
    auto_queries,
    evaluate_rankings,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    search,
)


def test_recall_counts_fraction_of_relevant_docs_in_top_k():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"a"}, 1) == 1.0
    assert recall_at_k(ranked, {"c"}, 2) == 0.0
    assert recall_at_k(ranked, {"a", "d"}, 3) == 0.5
    assert recall_at_k(ranked, {"a", "d"}, 4) == 1.0


def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank(["a", "b", "c"], {"c", "b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0


def test_ndcg_binary_relevance():
    # 정답 하나가 2위: DCG = 1/log2(3), IDCG = 1
    assert ndcg_at_k(["a", "b"], {"b"}, 2) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k(["b", "a"], {"b"}, 2) == 1.0
    assert ndcg_at_k(["a", "b"], {"b"}, 1) == 0.0
    # 정답이 k보다 많으면 IDCG는 k개까지만 센다
    assert ndcg_at_k(["a", "b"], {"a", "b", "c"}, 2) == 1.0


def test_search_ranks_documents_not_chunks():
    # doc1의 청크 두 개가 질의와 가장 가깝지만, 결과에는 doc1이 한 번만 나와야 한다
    matrix = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    doc_ids = ["doc1", "doc1", "doc2"]
    ranked = search(np.array([[1.0, 0.0]], dtype=np.float32), matrix, doc_ids, top_k=5)
    assert ranked == [["doc1", "doc2"]]


def test_search_top_k_limits_results():
    matrix = np.eye(3, dtype=np.float32)
    ranked = search(np.eye(3, dtype=np.float32), matrix, ["a", "b", "c"], top_k=1)
    assert ranked == [["a"], ["b"], ["c"]]


def test_evaluate_rankings_averages_metrics_and_collects_misses():
    queries = [
        EvalQuery(query="q1", relevant_doc_ids=["a"]),
        EvalQuery(query="q2", relevant_doc_ids=["z"]),
    ]
    rankings = [["a", "b"], ["a", "b"]]
    result = evaluate_rankings(queries, rankings, k_values=[1, 2])
    assert result["metrics"]["recall@1"] == 0.5
    assert result["metrics"]["recall@2"] == 0.5
    assert result["metrics"]["mrr"] == 0.5
    assert result["metrics"]["ndcg@2"] == 0.5
    assert [m["query"] for m in result["misses"]] == ["q2"]


def test_eval_query_requires_relevant_docs():
    with pytest.raises(ValidationError):
        EvalQuery(query="q", relevant_doc_ids=[])


def _chunk(doc_id, index, text, title):
    return {"chunk_id": f"{doc_id}-{index:03d}", "doc_id": doc_id, "chunk_index": index, "text": text, "metadata": {"title": title}}


CHUNKS = [
    _chunk("d1", 0, "Dense retrieval. Dense vectors find passages. More text here.", "Dense retrieval"),
    _chunk("d1", 1, "More text here. Tail sentence.", "Dense retrieval"),
    _chunk("d2", 0, "Sparse retrieval. BM25 counts terms well. Other text.", "Sparse retrieval"),
]


def test_auto_queries_title_mode_uses_one_query_per_document():
    queries, skipped = auto_queries(CHUNKS, mode="title", max_queries=10, seed=0)
    assert skipped == 0
    assert sorted((q.query, tuple(q.relevant_doc_ids)) for q in queries) == [
        ("Dense retrieval", ("d1",)),
        ("Sparse retrieval", ("d2",)),
    ]


def test_auto_queries_first_sentence_mode_strips_title_prefix():
    queries, _ = auto_queries(CHUNKS, mode="first_sentence", max_queries=10, seed=0)
    assert {q.relevant_doc_ids[0]: q.query for q in queries} == {
        "d1": "Dense vectors find passages.",
        "d2": "BM25 counts terms well.",
    }


def test_auto_queries_counts_documents_without_first_sentence():
    # 초록 첫 문장이 chunk_size보다 길면 글자 단위로 잘려 0번 청크에 제목만 남는다
    title_only = [_chunk("d3", 0, "Long abstract paper.", "Long abstract paper")]
    queries, skipped = auto_queries(CHUNKS + title_only, mode="first_sentence", max_queries=10, seed=0)
    assert len(queries) == 2
    assert skipped == 1


def test_auto_queries_samples_deterministically():
    first, _ = auto_queries(CHUNKS, mode="title", max_queries=1, seed=42)
    again, _ = auto_queries(CHUNKS, mode="title", max_queries=1, seed=42)
    assert len(first) == 1
    assert first == again
