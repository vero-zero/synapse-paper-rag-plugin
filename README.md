# Paper RAG Dataset — Synapse Plugin

[![test](https://github.com/vero-zero/synapse-paper-rag-plugin/actions/workflows/test.yml/badge.svg)](https://github.com/vero-zero/synapse-paper-rag-plugin/actions/workflows/test.yml)
![python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![synapse-sdk](https://img.shields.io/badge/synapse--sdk-v2-315efb)
![tests](https://img.shields.io/badge/tests-56%20passed-4ade80)

> **오픈액세스 논문을 수집해 RAG(검색 증강 생성)에 바로 쓸 수 있는 청크 데이터셋과 데이터 품질 리포트로 바꾸는 [Synapse](https://www.datamaker.io) 플러그인입니다.**
> 데이터메이커가 공개한 [`synapse-sdk`](https://pypi.org/project/synapse-sdk/) v2의 플러그인 구조(config.yaml · `BaseAction` · Storage)를 그대로 따랐습니다.

![pipeline](docs/images/pipeline.png)

<br>

## 프로젝트 목적

AI가 문서를 근거로 답하게 하려면(RAG), 문서를 **검색하기 좋은 크기로 자르고, 어디서 온 내용인지 함께 남겨야** 합니다. 그런데 논문 데이터는 그대로 쓰기 어렵습니다.

| 문제 | 이 플러그인의 처리 |
|---|---|
| 초록 길이가 제각각이라 검색 품질이 들쭉날쭉함 | 문장 경계를 지키며 정해진 길이로 분할 |
| 청크 경계에서 문맥이 끊김 | 앞 청크 끝을 겹쳐(overlap) 연결 |
| 청크 하나만 검색되면 어느 논문인지 모름 | 제목을 본문 앞에 붙이고 출처 메타데이터(DOI · 저자 · 연도 · 라이선스) 저장 |
| 같은 논문이 여러 번 수집됨 | DOI(없으면 정규화한 제목) 기준 중복 제거 |
| 몇 건이 왜 빠졌는지 알 수 없음 | 제외 사유 · 언어 분포 · 평균 길이를 품질 리포트로 기록 |
| 만든 데이터셋이 검색에 실제로 쓸 만한지 모름 | 임베딩 후 벡터 검색을 돌려 Recall@k · MRR · nDCG@k를 리포트로 기록 |

실무에서는 공공기관 연구자 포털에 논문 검색(ScienceON)과 해외 원문복사(RapidILL)를 연동해 **연구자에게 논문을 전달하는 일**을 했습니다. 이 프로젝트는 같은 논문 데이터를 **AI가 검색해 쓸 수 있는 데이터**로 만드는 쪽을 직접 해 보기 위해 시작했습니다.

<br>

## 실행 화면

### 1. 수집 → RAG 데이터셋 생성 (Synapse 계정 없이 로컬 실행)
영어 논문 30편을 수집해 청크 94개를 만든 실제 실행 결과입니다.

![run](docs/images/run.png)

### 2. 결과물 — 청크 1건과 품질 리포트
청크마다 임베딩할 텍스트와 출처 메타데이터가 함께 들어가고, `manifest.json`에 품질 지표가 남습니다.

![sample](docs/images/sample.png)

### 3. 청크 분할 동작
`chunk_size`와 `overlap`에 따라 청크가 어떻게 나뉘는지 보여 줍니다. 노란색은 앞 청크와 겹친 부분이고, 문장 중간이나 단어 중간에서 자르지 않습니다.

![chunk demo](docs/images/chunk_demo.gif)

### 4. 테스트 — 56개 통과 (GitHub Actions: Ubuntu · Windows)

![pytest](docs/images/pytest.png)

<br>

## 주요 기능

| 액션 | 하는 일 |
|---|---|
| `collect` | OpenAlex에서 **오픈액세스이면서 초록이 있는 논문**만 수집. 언어 · 출판연도 필터, 페이지네이션, 429/5xx 재시도(지수 백오프), 역색인 초록 복원 |
| `build_rag` | 중복 제거 → 짧거나 없는 초록 제외 → 문장 단위 청크 + overlap → 출처 메타데이터 부착 → `chunks.jsonl` · `manifest.json` 생성, Synapse 메트릭 기록 |
| `embed` | `chunks.jsonl`을 임베딩해 `embeddings.npy`(L2 정규화 float32) · `embedding_ids.json` · `embeddings_manifest.json` 생성. 백엔드는 `hash`(오프라인 기준선) 또는 `openai_compatible`(OpenAI · Ollama · vLLM · TEI) |
| `evaluate` | 저장된 벡터로 코사인 검색 → 문서 단위로 Recall@k · MRR · nDCG@k 계산 → `eval_report.json`. 정답셋 파일(`queries_file`)이 없으면 제목 · 초록 첫 문장으로 질의를 자동 생성 |

**실행 결과**
- 영어 "retrieval augmented generation" 30편 → 청크 94개 (평균 456자, 2026-11-11 실행)
- 한국어 "인공지능 교육" 20편 → 청크 35개
- 위 영어 30편을 `hash` 임베딩(256차원)으로 평가 (2026-11-13 실행)

  | 자동 질의 | 질의 수 | Recall@1 | Recall@10 | MRR |
  |---|---|---|---|---|
  | 초록 첫 문장 | 26 (4편은 건너뜀) | 0.96 | 1.00 | 0.98 |
  | 제목 | 30 | 0.33 | 0.90 | 0.49 |

  첫 문장 질의는 그 문장이 청크에 그대로 들어 있어 점수가 거의 만점입니다(누설). 제목 질의는 30편이 모두 RAG 논문이라 제목 단어가 서로 겹쳐, 단어 겹침만 보는 해시 임베딩으로는 구분이 어렵습니다. 그래서 자동 질의 결과는 임베딩 설정끼리 비교하는 기준선으로만 쓰고, 실제 품질은 정답셋으로 확인하도록 리포트에 경고를 남깁니다.

<br>

## 빠른 시작

```bash
# synapse-sdk는 Python 3.12 이상 필요
uv venv --python 3.12 && uv pip install -e . pytest respx

# 수집 → RAG 데이터셋 생성 (Synapse 계정 없이 로컬에서 실행)
synapse plugin run collect   --mode local --params '{"query": "retrieval augmented generation", "language": "en", "max_papers": 30, "dataset_name": "demo"}'
synapse plugin run build_rag --mode local --params '{"dataset_name": "demo", "chunk_size": 600, "chunk_overlap": 80}'

# 임베딩 → 검색 품질 평가 (hash는 API 키 없이 동작하는 기준선)
synapse plugin run embed    --mode local --params '{"dataset_name": "demo"}'
synapse plugin run evaluate --mode local --params '{"dataset_name": "demo", "k_values": [1, 5, 10]}'

# 실제 임베딩 모델 (OpenAI 호환 API). 키는 EMBEDDING_API_KEY 환경변수로 넘기는 것을 권장
#   embed    --params '{"dataset_name": "demo", "provider": "openai_compatible", "model": "text-embedding-3-small"}'
# 정답셋으로 평가: output/demo/gold.jsonl 한 줄 = {"query": "...", "relevant_doc_ids": ["..."]}
#   evaluate --params '{"dataset_name": "demo", "queries_file": "gold.jsonl"}'

# S3 · MinIO에 저장하려면 storage 파라미터만 바꾼다
#   "storage": {"provider": "s3", "configuration": {"bucket_name": "...", "endpoint_url": "..."}}

pytest   # 56개
```

<br>

## 설계 포인트

- **로직과 플랫폼을 분리했습니다.** 수집(`plugin/core/openalex.py`)과 청크(`plugin/core/chunking.py`)는 SDK를 import하지 않는 순수 Python이고, Synapse 액션(`plugin/collect.py`, `plugin/build_rag.py`)은 파라미터 검증 · 진행률 · 메트릭 · 저장만 맡습니다. 그래서 핵심 로직은 SDK 없이 단위 테스트합니다. 저장을 맡는 `plugin/core/io.py`만 SDK의 Storage를 씁니다.
- **잘못된 요청은 실행 전에 거절합니다.** Pydantic으로 `chunk_overlap < chunk_size`를 검사하고, `dataset_name`은 `../` 같은 경로 조작을 막습니다.
- **저장소를 바꿔도 코드는 그대로입니다.** 결과는 SDK의 `get_storage()`로 저장해, `storage` 파라미터만 바꾸면 로컬 대신 S3 · MinIO · GCS · SFTP에 저장됩니다.
- **데이터 품질을 드러냅니다.** 빠진 건수와 이유, 언어 분포, 평균 청크 길이를 `manifest.json`과 `set_metrics`로 남깁니다.
- **평가가 엉뚱한 벡터로 돌지 않게 막습니다.** `embed`는 `chunks.jsonl`의 sha256과 모델 정보를 매니페스트에 남기고, `evaluate`는 청크가 바뀌었으면 실패시키며 질의도 매니페스트에 적힌 같은 모델로 임베딩합니다. API 키는 매니페스트 · 리포트에 남기지 않습니다.
- **정답은 청크가 아니라 논문 단위로 판정합니다.** 한 논문의 청크가 상위권을 여러 칸 차지해도 같은 논문 하나로 세야 사용자가 체감하는 검색 품질과 맞습니다.

```
plugin/
├── collect.py        # Synapse 액션: 파라미터 검증 · 진행률 · 저장
├── build_rag.py      # Synapse 액션: 청크 생성 · 품질 리포트 · 메트릭
├── embed.py          # Synapse 액션: 임베딩 생성 · 매니페스트
├── evaluate.py       # Synapse 액션: 벡터 검색 · 품질 지표 리포트
└── core/             # 핵심 로직 (io 외에는 SDK 없이 단위 테스트)
    ├── openalex.py   # API 클라이언트 · 재시도 · 초록 복원
    ├── chunking.py   # 문장 분할 · overlap · 중복 제거 · 품질 리포트
    ├── embedding.py  # hash · OpenAI 호환 임베더 (배치 · 재시도 · 차원 검증)
    ├── retrieval.py  # 코사인 검색 · Recall@k · MRR · nDCG@k · 자동 질의
    └── io.py         # JSONL 입출력 · Storage 업로드 (SDK 사용)
```

<br>

## 기술 스택과 선택 이유

| 기술 | 쓰는 곳 | 선택 이유 |
|---|---|---|
| synapse-sdk v2 | 플러그인 구조 · 저장 · 진행률 · 메트릭 | Synapse 플랫폼에 그대로 올릴 수 있는 형식. 공식 CLI(`synapse plugin create`)로 뼈대 생성 |
| OpenAlex API | 논문 수집 | 무료 공개 API이고 오픈액세스 여부 · 초록 · 라이선스 정보를 함께 제공. 단, 공개 접근이 재배포 허가를 뜻하지는 않으므로 라이선스는 청크마다 `metadata.license`로 남기고 사용 판단은 데이터셋을 쓰는 쪽에 맡김 |
| httpx + 지수 백오프 | API 호출 | 429 · 5xx 같은 일시 오류는 재시도하고, 4xx는 즉시 실패시켜 잘못된 요청을 반복하지 않음 |
| Pydantic v2 | 액션 파라미터 | 잘못된 설정을 실행 전에 거절 |
| JSONL | 결과 파일 | 한 줄에 한 건이라 다른 도구가 줄 단위로 읽을 수 있고, RAG 도구가 바로 읽음. 현재 build_rag는 전체를 메모리에 올려 처리(수집 상한 1,000건 기준) |
| 글자 수 기준 청크 | 청크 분할 | 특정 임베딩 모델의 토크나이저에 묶이지 않게. 대신 토큰 수는 보장하지 않으므로 임베딩 전에 대상 모델 기준으로 확인 필요. 문장 분리는 공백으로 문장을 구분하는 언어(영어 · 한국어 등) 기준 |
| numpy brute-force 검색 | 벡터 검색 평가 | 수만 청크까지는 행렬곱 한 번이면 충분하고, 벡터 DB 없이 평가 결과를 재현할 수 있음. 그 이상 규모는 벡터 DB의 몫 |
| 해시 임베딩 기준선 | `embed`의 기본값 | 의존성 · API 키 없이 항상 같은 결과를 내서 CI와 오프라인 데모에 사용. 의미는 모르고 단어 겹침만 보므로, 실제 모델과 비교할 기준선 역할 |
| pytest · respx · GitHub Actions | 테스트 · CI | 외부 API 없이 재현 가능한 테스트, Ubuntu · Windows 양쪽 자동 실행 |

**범위에서 뺀 것:** 원문 PDF는 라이선스와 OCR 비용 때문에 1차 범위에서 제외하고 초록만 사용했습니다.

<br>

## 테스트

| 파일 | 내용 |
|---|---|
| [tests/test_chunking.py](tests/test_chunking.py) | 청크 최대 길이, 문장 경계, overlap 연결, 긴 문장 분할, 중복 제거(다국어 제목 포함), 품질 리포트, 메타데이터 |
| [tests/test_openalex.py](tests/test_openalex.py) | 초록 복원, 응답 정규화, 페이지네이션, 필터, 429 재시도, Retry-After 대기(비정상 값 포함), 장애 시 예외, 4xx는 재시도하지 않음 |
| [tests/test_actions.py](tests/test_actions.py) | `config.yaml`의 액션 탐색, SDK `LocalExecutor`로 collect → build_rag 전체 실행, 파라미터 검증 실패 |
| [tests/test_retrieval.py](tests/test_retrieval.py) | Recall@k · MRR · nDCG@k 손계산 값, 논문 단위 검색, 자동 질의 생성(제목 접두어 제거 · 건너뛴 문서 수 · 시드 고정) |
| [tests/test_embedding.py](tests/test_embedding.py) | 해시 임베딩 결정성 · 정규화, OpenAI 호환 배치 · 응답 순서 복원 · 429 재시도 · 4xx 즉시 실패 · 차원 · 개수 불일치 |
| [tests/test_embed_evaluate.py](tests/test_embed_evaluate.py) | build_rag → embed → evaluate 전체 실행, 정답셋 파일, 청크가 바뀐 뒤 평가 거부, 경로 조작 차단, API 키가 결과 파일에 남지 않음 |

<br>

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

### 3. JS로 옮긴 청크 로직이 Python 원본과 1건 불일치
- **증상:** 시연용으로 옮긴 청크 로직을 15개 조건으로 비교하자 1건에서 청크 앞 공백이 다름
- **원인:** overlap이 0이고 조각이 공백으로 시작할 때 Python은 `.strip()` 경로를 타지만 JS는 다른 분기로 빠짐
- **해결:** 분기 조건을 원본과 같게 맞춤 → 15/15 일치

### 4. 플러그인이 허용한 Python 3.10에서 build_rag import 실패
- **증상:** 코드 리뷰에서 `config.yaml`은 Python `>=3.10`을 허용하는데 `build_rag.py`가 3.10에서 import되지 않는다는 지적을 받음
- **원인:** `datetime.UTC`는 Python 3.11부터 제공됨. 게다가 `synapse-sdk` 자체가 `Requires-Python: >=3.12`라 3.10 · 3.11에서는 SDK 설치부터 불가능. `pyproject.toml` · CI · README는 3.12였는데 `config.yaml`만 달랐음
- **해결:** 코드를 3.10에 맞추지 않고, `config.yaml`을 실제 최소 버전인 `>=3.12`로 맞춤
- **재발 방지:** Python 버전은 SDK의 요구 버전을 기준으로 `config.yaml` · `pyproject.toml` · CI에 같은 값으로 선언

### 5. DOI 없는 일본어 · 러시아어 논문이 모두 중복으로 제거됨
- **증상:** DOI가 없는 논문 중 제목이 일본어 · 키릴 문자로만 된 논문끼리 서로 다른 논문인데도 하나만 남음
- **원인:** 제목 중복 키가 영문 · 숫자 · 한글(`[^0-9a-z가-힣]` 제거)만 남겨서, 그 외 문자로 된 제목은 키가 빈 값(`title:`)으로 같아짐
- **해결:** 유니코드 글자는 모두 남기고 구두점 · 공백만 제거. NFKC 정규화와 `casefold`로 전각/반각 · 대소문자 차이를 맞추고, 제목이 구두점뿐이면 OpenAlex ID로 구분
- **재발 방지:** 일본어 · 키릴 문자 · 악센트 문자 · 전각 문자 · 구두점만 있는 제목을 테스트에 추가. 단, 이 변경으로 해당 논문의 `doc_id`가 바뀌므로 기존 데이터셋은 다시 생성해야 함

### 6. 429 응답의 Retry-After를 무시하고, 비정상 값이면 수집이 중단될 수 있음
- **증상:** OpenAlex가 429와 함께 `Retry-After`로 대기 시간을 알려줘도 1 · 2 · 4초 간격으로만 재시도. 이를 반영한 뒤에는 `Retry-After: NaN`이 오면 수집 작업 전체가 예외로 중단될 수 있다는 리뷰를 받음
- **원인:** 재시도 대기를 고정된 지수 백오프로만 계산. 반영 후에는 `float("NaN")`이 오류 없이 숫자로 변환되고 모든 비교를 통과해 `time.sleep(nan)`에서 `ValueError` 발생
- **해결:** 초 단위 `Retry-After`를 우선 따르되 최대 60초로 제한. 값이 없거나 날짜 형식 · `NaN` · `inf` · 음수이면 기존 지수 백오프 사용
- **재발 방지:** 초 단위 · 상한 초과 · 날짜 형식 · NaN · inf · 음수 6가지 경우를 테스트로 고정

### 7. 초록 첫 문장 자동 질의가 30편 중 26편만 생성됨
- **증상:** 실제 데이터로 `evaluate`를 돌리자 질의가 26개만 만들어졌는데, 리포트에는 빠진 문서 수가 0으로 나옴
- **원인:** 초록 첫 문장이 `chunk_size`보다 길면 청커가 그 문장을 글자 단위로 잘라 다음 청크로 넘김 → 0번 청크에 `제목.`만 남아 첫 문장을 찾을 수 없음. 이 경우를 조용히 건너뛰고 있었음
- **해결:** 잘린 조각을 억지로 질의로 쓰지 않고 건너뛰되, 건너뛴 문서 수를 `skipped_queries`로 리포트에 남김
- **재발 방지:** 0번 청크에 제목만 있는 경우를 테스트로 고정

<br>

## 개발 방식

Synapse SDK를 사용해 OpenAlex 논문을 수집하고, 출처와 품질 정보를 유지하면서 RAG용 데이터셋으로 변환하는 플러그인을 만들었습니다. Claude Code를 구현에 활용했지만, 제가 문제 범위와 검증 기준을 정하고 생성된 코드를 테스트와 리뷰로 확인했습니다. 그 과정에서 버전 조건, 다국어 중복 키, API 재시도 동작을 발견해 수정했습니다. 이후 임베딩 생성과 벡터 검색 품질 평가까지 범위를 넓혀, 만든 데이터셋이 검색에 쓸 만한지 수치로 확인할 수 있게 했습니다.

## 다음 단계

- [x] 임베딩 생성과 벡터 검색 품질 평가 액션 추가
- [ ] 실제 임베딩 모델과 정답셋으로 hash 기준선 대비 품질 비교
- [ ] FastAPI 검색 API로 "검색해서 출처와 함께 돌려주는" 단계 연결
- [ ] 오픈액세스 PDF 원문 청크 (OCR 포함)
- [ ] Synapse `BaseUploadAction`으로 Synapse 데이터셋에 바로 업로드

<br>

---

**만든 사람** · 고테아 · [포트폴리오](https://vero-zero.github.io) · [GitHub](https://github.com/vero-zero)
