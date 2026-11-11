import httpx
import pytest
import respx

from plugin.core.openalex import OpenAlexClient, SourceUnavailable, reconstruct_abstract, to_paper

BASE = "https://openalex.test"


def work(i: int) -> dict:
    return {
        "id": f"https://openalex.org/W{i}",
        "doi": f"https://doi.org/10.1/{i}",
        "title": f"Paper {i}",
        "publication_year": 2024,
        "language": "en",
        "authorships": [{"author": {"display_name": "Kim"}}, {"author": {}}],
        "primary_location": {"source": {"display_name": "Journal"}, "license": "cc-by"},
        "open_access": {"is_oa": True, "oa_url": f"https://oa.test/{i}.pdf"},
        "abstract_inverted_index": {"RAG": [0], "works": [1], "well": [2]},
        "cited_by_count": 3,
    }


def test_reconstruct_abstract_from_inverted_index():
    assert reconstruct_abstract({"world": [1], "hello": [0], "again": [3], "hello2": [2]}) == "hello world hello2 again"
    assert reconstruct_abstract(None) is None


def test_to_paper_normalizes_fields():
    paper = to_paper(work(7))
    assert paper.source_id == "W7"
    assert paper.doi == "10.1/7"
    assert paper.abstract == "RAG works well"
    assert paper.authors == ["Kim"]
    assert paper.license == "cc-by"
    assert paper.oa_url == "https://oa.test/7.pdf"


@respx.mock
def test_iter_papers_paginates_and_stops_at_max():
    route = respx.get(f"{BASE}/works").mock(
        side_effect=[
            httpx.Response(200, json={"meta": {"count": 5}, "results": [work(1), work(2)]}),
            httpx.Response(200, json={"meta": {"count": 5}, "results": [work(3), work(4)]}),
        ]
    )
    with httpx.Client() as http:
        client = OpenAlexClient(http, base_url=BASE, page_size=2)
        got = [p.source_id for p, _ in client.iter_papers("rag", max_papers=3)]
    assert got == ["W1", "W2", "W3"]
    assert route.call_count == 2
    sent = route.calls[0].request.url.params
    assert sent["filter"] == "is_oa:true,has_abstract:true"


@respx.mock
def test_filters_include_language_and_year():
    route = respx.get(f"{BASE}/works").mock(return_value=httpx.Response(200, json={"meta": {"count": 0}, "results": []}))
    with httpx.Client() as http:
        list(OpenAlexClient(http, base_url=BASE).iter_papers("rag", max_papers=5, language="ko", from_year=2020))
    assert route.calls[0].request.url.params["filter"] == "is_oa:true,has_abstract:true,language:ko,from_publication_date:2020-01-01"


@respx.mock
def test_retries_on_429_then_succeeds():
    route = respx.get(f"{BASE}/works").mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json={"meta": {"count": 1}, "results": [work(1)]})]
    )
    with httpx.Client() as http:
        client = OpenAlexClient(http, base_url=BASE, backoff_seconds=0)
        assert len(list(client.iter_papers("rag", max_papers=1))) == 1
    assert route.call_count == 2


@respx.mock
@pytest.mark.parametrize(("header", "expected"), [("2", 2.0), ("9999", 60.0), ("Wed, 21 Oct 2026 07:28:00 GMT", 5.0)])
def test_429_waits_for_retry_after(monkeypatch, header, expected):
    waits: list[float] = []
    monkeypatch.setattr("plugin.core.openalex.time.sleep", waits.append)
    respx.get(f"{BASE}/works").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": header}),
            httpx.Response(200, json={"meta": {"count": 1}, "results": [work(1)]}),
        ]
    )
    with httpx.Client() as http:
        list(OpenAlexClient(http, base_url=BASE, backoff_seconds=5).iter_papers("rag", max_papers=1))
    assert waits == [expected]  # 초 단위는 따르고, 너무 길면 상한, 날짜 형식이면 지수 백오프


@respx.mock
def test_raises_source_unavailable_after_retries():
    respx.get(f"{BASE}/works").mock(side_effect=httpx.ConnectError("down"))
    with httpx.Client() as http:
        client = OpenAlexClient(http, base_url=BASE, max_retries=2, backoff_seconds=0)
        with pytest.raises(SourceUnavailable):
            list(client.iter_papers("rag", max_papers=1))


@respx.mock
def test_client_error_is_not_retried():
    route = respx.get(f"{BASE}/works").mock(return_value=httpx.Response(400))
    with httpx.Client() as http:
        with pytest.raises(httpx.HTTPStatusError):
            list(OpenAlexClient(http, base_url=BASE, backoff_seconds=0).iter_papers("rag", max_papers=1))
    assert route.call_count == 1
