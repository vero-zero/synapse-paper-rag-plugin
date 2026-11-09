# Paper RAG Dataset – Synapse Plugin

[![test](https://github.com/vero-zero/synapse-paper-rag-plugin/actions/workflows/test.yml/badge.svg)](https://github.com/vero-zero/synapse-paper-rag-plugin/actions/workflows/test.yml)

오픈액세스 논문을 수집해 **RAG(검색 증강 생성)용 청크 데이터셋**과 **데이터 품질 리포트**를 만드는 [Synapse](https://www.datamaker.io) 플러그인입니다.
데이터메이커가 공개한 [`synapse-sdk`](https://pypi.org/project/synapse-sdk/) v2의 플러그인 구조(config.yaml, `BaseAction`, Storage)를 따라 만들었습니다.

```
synapse plugin run collect   →  papers.jsonl      (논문 메타데이터 + 초록)
synapse plugin run build_rag →  chunks.jsonl      (임베딩용 텍스트 + 출처 메타데이터)
                                manifest.json     (파라미터, 건수, 제외 사유, 언어 분포)
```

## 왜 만들었나

공공기관 연구자 포털에서 ScienceON 논문 검색과 해외 원문복사(RapidILL)를 연동하고 운영하면서, 연구자에게 **논문을 전달하는 일**을 했습니다.
이 플러그인은 같은 논문 데이터를 **AI가 쓸 수 있는 데이터**로 만드는 쪽을 다룹니다.
데이터를 모으고, 정제하고, AI가 검색해 쓸 수 있는 형태로 만드는 과정을 Synapse 플러그인 하나로 묶었습니다.

## 액션

| 액션 | 실행 방식 | 하는 일 |
|---|---|---|
| `collect` | task | OpenAlex에서 오픈액세스이고 초록이 있는 논문만 수집. 언어·출판연도 필터, 페이지네이션, 429/5xx 재시도 |
| `build_rag` | task | DOI(없으면 정규화한 제목) 기준 중복 제거 → 짧거나 없는 초록 제외 → 문장 경계를 지키는 청크 분할(overlap 포함) → 출처 메타데이터 부착 → 품질 리포트 생성 |

### 출력 예시 (`chunks.jsonl` 한 줄)

```json
{
  "chunk_id": "ac60bdefec66e3bd-000",
  "doc_id": "ac60bdefec66e3bd",
  "text": "Graph Retrieval-Augmented Generation: A Survey. Recently, Retrieval-Augmented Generation (RAG) has achieved ...",
  "chunk_index": 0,
  "chunk_count": 3,
  "metadata": {"title": "...", "doi": "10.1145/3777378", "year": 2025, "authors": ["..."], "venue": "...", "license": "...", "oa_url": "..."}
}
```

실제로 실행한 결과입니다.
- 영어 "retrieval augmented generation" 30편 → 청크 94개 (평균 451자)
- 한국어 "인공지능 교육" 20편 → 청크 35개

## 설계 포인트

- **로직과 플랫폼을 분리했습니다.** `plugin/core/`(수집, 청크, 입출력)는 순수 Python이고, Synapse 액션(`plugin/collect.py`, `plugin/build_rag.py`)은 파라미터 검증, 진행률, 메트릭, 저장만 맡습니다. 그래서 핵심 로직은 SDK 없이 단위 테스트합니다.
- **파라미터는 Pydantic으로 검증합니다.** 예를 들어 `chunk_overlap < chunk_size`를 검사하고, `dataset_name`은 `../` 같은 경로 조작을 막습니다. 잘못된 요청은 실행 전에 거절됩니다.
- **저장소를 바꿔도 코드는 그대로입니다.** 결과는 SDK의 `get_storage()`로 저장합니다. `storage` 파라미터만 바꾸면 로컬 대신 S3·MinIO·GCS·SFTP에 저장됩니다.
- **데이터 품질을 드러냅니다.** 몇 건이 왜 빠졌는지(중복, 초록 부족), 언어 분포, 평균 청크 길이를 `manifest.json`과 Synapse 메트릭(`set_metrics`)으로 남깁니다.
- **청크 품질을 챙겼습니다.**
  - 제목을 본문 앞에 붙여서, 청크 하나만 검색돼도 어느 논문인지 알 수 있게 했습니다.
  - 앞 청크 끝을 겹쳐서 경계에 걸친 문맥이 끊기지 않게 했습니다.
  - 겹치는 부분이 단어 중간에서 시작하지 않게 했습니다.
- **OpenAlex 초록 복원:** OpenAlex는 초록을 `{단어: [위치]}` 역색인으로 줍니다. 이를 원문 순서로 복원합니다.

## 기술 스택과 선택 이유

| 기술 | 쓰는 곳 | 선택 이유 |
|---|---|---|
| synapse-sdk v2 | 플러그인 구조, 저장, 진행률·메트릭 | Synapse 플랫폼에 그대로 올릴 수 있는 형식. 공식 CLI(`synapse plugin create`)로 뼈대 생성 |
| OpenAlex API | 논문 수집 | 무료 공개 API이고 오픈액세스 여부·초록을 제공해 라이선스 걱정 없이 사용 가능 |
| httpx + 지수 백오프 | API 호출 | 429·5xx 같은 일시 오류는 재시도하고, 4xx는 즉시 실패시켜 잘못된 요청을 반복하지 않음 |
| Pydantic v2 | 액션 파라미터 | 잘못된 설정(겹침 ≥ 청크 길이, 경로 조작)을 실행 전에 거절 |
| JSONL | 결과 파일 | 한 줄에 한 건이라 대용량도 스트리밍 처리 가능, RAG 도구가 바로 읽음 |
| 글자 수 기준 청크 | 청크 분할 | 특정 임베딩 모델의 토크나이저에 묶이지 않게. 모델 선택은 데이터셋을 쓰는 쪽에 맡김 |
| pytest · respx · GitHub Actions | 테스트 · CI | 외부 API 없이 재현 가능한 테스트, Ubuntu·Windows 양쪽 자동 실행 |

**범위에서 뺀 것:** 원문 PDF는 라이선스와 OCR 비용 때문에 1차 범위에서 제외하고 초록만 사용했습니다.

## 데모

- 청크 분할 로직(`chunk_text()`)을 JavaScript로 옮긴 데모를 [포트폴리오](https://vero-zero.github.io/#demo)에서 직접 돌려 볼 수 있습니다. Python 원본과 15개 조건에서 같은 결과가 나오는 것을 확인했습니다.

## 실행

```bash
uv venv --python 3.12 && uv pip install -e . pytest respx   # synapse-sdk는 Python 3.12 이상 필요

# 수집 → RAG 데이터셋 생성 (Synapse 계정 없이 로컬에서 실행)
synapse plugin run collect   --mode local --params '{"query": "retrieval augmented generation", "language": "en", "max_papers": 30, "dataset_name": "demo"}'
synapse plugin run build_rag --mode local --params '{"dataset_name": "demo", "chunk_size": 600, "chunk_overlap": 80}'

# S3/MinIO에 저장하려면 storage 파라미터만 바꾼다
#   "storage": {"provider": "s3", "configuration": {"bucket_name": "...", "endpoint_url": "..."}}

pytest   # 19개 (GitHub Actions에서 Ubuntu·Windows로 push마다 자동 실행)
```

## 테스트

| 파일 | 내용 |
|---|---|
| [tests/test_chunking.py](tests/test_chunking.py) | 청크 최대 길이, 문장 경계, overlap 연결, 긴 문장 분할, 중복 제거, 품질 리포트, 메타데이터 |
| [tests/test_openalex.py](tests/test_openalex.py) | 초록 복원, 응답 정규화, 페이지네이션, 필터, 429 재시도, 장애 시 예외, 4xx는 재시도하지 않음 |
| [tests/test_actions.py](tests/test_actions.py) | `config.yaml`의 액션 탐색, SDK `LocalExecutor`로 collect → build_rag 전체 실행, 파라미터 검증 실패 |

## 트러블슈팅

### 1. 한국어 Windows에서만 플러그인 탐색 실패
- **증상:** `config.yaml`에 한글 설명을 넣자 `UnicodeDecodeError`
- **원인:** `synapse-sdk 2026.2.22`가 `config.yaml`을 인코딩 지정 없이 엶(`plugins/discovery.py`의 `path.open()`) → 한국어 Windows에서는 cp949로 읽음
- **해결:** `config.yaml`은 ASCII로만 작성. SDK에서 `encoding="utf-8"`을 지정하면 근본 해결
- **재발 방지:** GitHub Actions에 Windows 러너를 추가해 OS별 문제를 함께 확인

### 2. 파라미터 검증 실패가 다른 예외로 올라옴
- **증상:** 잘못된 파라미터 테스트에서 `ValidationError`를 기대했는데 다른 예외가 발생
- **원인:** SDK `LocalExecutor`가 검증 오류를 `ExecutionError`로 감싸서 전달
- **해결:** 통합 테스트는 `ExecutionError`와 메시지로 검증하고, 순수 로직은 SDK 없이 단위 테스트

### 3. JS 이식본과 Python 원본 결과 1건 불일치
- **증상:** 데모용으로 옮긴 청크 로직을 15개 조건으로 비교하자 1건에서 청크 앞 공백이 다름
- **원인:** overlap이 0이고 조각이 공백으로 시작할 때 Python은 `.strip()` 경로를 타지만 JS는 다른 분기로 빠짐
- **해결:** 분기 조건을 원본과 같게 맞춤 → 15/15 일치

## 개발 방식

방향과 도메인 설계, 검증 기준은 직접 정했습니다. 구현과 테스트 작성은 AI 코딩 에이전트(Claude Code)와 함께 진행했습니다.

## 다음 단계 (아이디어)

- 오픈액세스 PDF 원문을 내려받아 본문까지 청크로 만들기 (OCR 포함)
- Synapse `BaseUploadAction`으로 결과를 Synapse 데이터셋에 바로 업로드하기
- 임베딩 생성과 벡터 검색 품질 평가 액션 추가
