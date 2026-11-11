import pytest

from plugin.core.chunking import build_chunks, chunk_text, dedupe_key
from plugin.core.openalex import Paper

SENTENCE = "Retrieval augmented generation grounds model answers in external documents."  # 76자


def make_paper(**kw) -> Paper:
    base = dict(source_id="W1", title="A Survey of RAG", abstract=" ".join([SENTENCE] * 10), doi="10.1/abc", language="en")
    return Paper(**(base | kw))


def test_chunks_never_exceed_chunk_size():
    chunks = chunk_text(" ".join([SENTENCE] * 30), chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunks_keep_sentence_boundaries():
    chunks = chunk_text(" ".join([SENTENCE] * 10), chunk_size=200, overlap=0)
    assert all(c.endswith(".") for c in chunks)


def test_overlap_repeats_tail_of_previous_chunk():
    chunks = chunk_text(" ".join(f"Sentence number {i} is here." for i in range(40)), chunk_size=120, overlap=40)
    for prev, nxt in zip(chunks, chunks[1:]):
        # 다음 청크의 첫 단어가 앞 청크 끝부분에 있어야 문맥이 이어진다
        assert nxt.split()[0] in prev[-40:]


def test_very_long_sentence_is_split_by_characters():
    chunks = chunk_text("x" * 1000, chunk_size=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(c for c in chunks).count("x") >= 1000


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=100, overlap=100)


def test_dedupe_by_doi_then_normalized_title():
    assert dedupe_key(make_paper(doi="10.1/ABC")) == dedupe_key(make_paper(doi="10.1/abc"))
    assert dedupe_key(make_paper(doi=None, title="A Survey of RAG!")) == dedupe_key(make_paper(doi=None, title="a survey of rag"))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("検索拡張生成の研究", "ラベル付きデータの研究"),  # 일본어
        ("Обзор методов", "Анализ данных"),  # 키릴 문자
        ("Café study", "Caf study"),  # 악센트 문자
    ],
    ids=["japanese", "cyrillic", "accented"],
)
def test_title_key_keeps_non_ascii_letters(a, b):
    assert dedupe_key(make_paper(doi=None, title=a)) != dedupe_key(make_paper(doi=None, title=b))


def test_title_key_normalizes_width_and_punctuation():
    assert dedupe_key(make_paper(doi=None, title="ＲＡＧ：検索")) == dedupe_key(make_paper(doi=None, title="rag 検索"))


def test_punctuation_only_titles_do_not_collide():
    a = make_paper(doi=None, title="???", source_id="W1")
    b = make_paper(doi=None, title="!!!", source_id="W2")
    assert dedupe_key(a) != dedupe_key(b)


def test_build_chunks_skips_duplicates_and_short_abstracts_with_report():
    papers = [
        make_paper(),
        make_paper(source_id="W2"),  # 같은 DOI → 중복
        make_paper(source_id="W3", doi="10.1/short", abstract="too short"),
        make_paper(source_id="W4", doi="10.1/none", abstract=None),
        make_paper(source_id="W5", doi="10.1/ko", language="ko"),
    ]
    chunks, report = build_chunks(papers, chunk_size=300, overlap=50, min_abstract_chars=100)

    assert report.input_papers == 5
    assert report.output_papers == 2
    assert report.skipped == {"duplicate": 1, "short_or_missing_abstract": 2}
    assert report.languages == {"en": 1, "ko": 1}
    assert report.output_chunks == len(chunks)


def test_chunk_metadata_and_ids():
    chunks, _ = build_chunks([make_paper()], chunk_size=300, overlap=50)
    first = chunks[0]
    assert first.text.startswith("A Survey of RAG.")  # 제목이 본문 앞에 붙는다
    assert first.metadata["doi"] == "10.1/abc"
    assert "abstract" not in first.metadata
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.chunk_count == len(chunks) for c in chunks)
