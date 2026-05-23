# Architecture

> GNN Product Recommender — LightGCN 기반 추천 시스템의 시스템 아키텍처 문서

본 문서는 코드 베이스의 구조, 런타임 컴포넌트, 데이터 흐름, 배포 토폴로지를 설명합니다.
UML 다이어그램(클래스/시퀀스/컴포넌트)은 [`uml.md`](./uml.md) 를 참조하세요.

---

## 1. 개요

### 1.1 도메인

- **목표**: 사용자-상품 상호작용 그래프로부터 사용자에게 Top-K 상품을 추천한다.
- **데이터셋**: Amazon Reviews 2023 — *All Beauty* 카테고리 (스킨/헤어/메이크업)
- **모델**: LightGCN (PyTorch Geometric 내장 구현, 3-layer GCN)
- **학습 목적함수**: BPR(Bayesian Personalized Ranking) + L2 정규화
- **평가**: Leave-one-out + Full-ranking 프로토콜 (HR@K, NDCG@K)

### 1.2 시스템 한눈에 보기

```
┌──────────────────────────── frontend (Vite + React + TS) ─────────────────────────┐
│                                                                                   │
│   App.tsx ── UserPicker ─┐                                                        │
│             ── HistoryList │── React Query ── /api/client.ts ──► HTTP fetch       │
│             ── RecommendationList                                                 │
│                                                                                   │
└────────────────────────────────────┬──────────────────────────────────────────────┘
                                     │ /api/*
┌────────────────────────────────────▼──────────────────────────────────────────────┐
│                        backend (FastAPI + PyTorch)                                │
│                                                                                   │
│   api/main.py  ──(lifespan)──► service.load()                                     │
│      │                              │                                             │
│      │                              ├── data.preprocess()  (Pandas + HF Hub)      │
│   api/routes.py ──► api/service.py ──┤                                            │
│      │                              ├── model.load_checkpoint() (PyG LightGCN)    │
│      │                              └── precompute user/item embeddings (CPU)     │
│      ▼                                                                            │
│   api/schemas.py (Pydantic)                                                       │
│                                                                                   │
│   ── offline ───────────────────────────────────────────────────────────────────  │
│   run.py ──► data.preprocess() ──► train.train() ──► checkpoints/best_model.pt    │
│                                          │                                        │
│                                          └── evaluate.compute_metrics()           │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 디렉터리 구조와 책임

```
gnn-recommender/
├── backend/                       # Python 3.10 — 학습 + 서빙
│   ├── api/                       # FastAPI HTTP 레이어
│   │   ├── main.py                #   app 생성, lifespan, SPA 정적 마운트
│   │   ├── routes.py              #   /api/* 엔드포인트 (얇은 컨트롤러)
│   │   ├── schemas.py             #   Pydantic 응답 모델 (DTO)
│   │   └── service.py             #   추론 서비스 — 모델/임베딩/추천 로직
│   ├── config.py                  # Config 데이터클래스 + 로깅 + 시드
│   ├── data.py                    # 데이터셋 다운로드/전처리/그래프 구성
│   ├── model.py                   # LightGCN 팩토리 + 체크포인트 I/O
│   ├── train.py                   # BPR 학습 루프, 음성 샘플링, ES, LR scheduler
│   ├── evaluate.py                # HR@K / NDCG@K full-ranking 평가
│   ├── run.py                     # CLI 오케스트레이터 (data → train)
│   ├── app_gradio.py              # (LEGACY) Gradio 단일 페이지 데모
│   ├── requirements.txt
│   └── data/                      # gitignore — raw / processed / checkpoints
│
├── frontend/                      # Node 18 — Vite + React + TS + Tailwind
│   ├── src/
│   │   ├── api/
│   │   │   ├── client.ts          # fetch 래퍼 + API 메서드 + 인라인 DTO 타입
│   │   │   └── types.ts           # `npm run typegen` 으로 OpenAPI에서 생성
│   │   ├── components/
│   │   │   ├── UserPicker.tsx     # 검색/페이지네이션/랜덤 사용자 선택
│   │   │   ├── HistoryList.tsx    # 선택 사용자의 학습 데이터 이력 표시
│   │   │   └── RecommendationList.tsx  # Top-K 추천 결과 표시
│   │   ├── App.tsx                # 3-컬럼 레이아웃, k 슬라이더, health 폴링
│   │   └── main.tsx               # React Query 프로바이더 + ReactDOM mount
│   ├── vite.config.ts             # 5173 dev 서버, /api → 8000 프록시
│   ├── tailwind.config.js
│   └── package.json
│
├── package.json                   # 루트 — concurrently 로 dev:api + dev:web 동시 기동
├── docs/                          # ← 본 문서
└── README.md
```

각 레이어의 단일 책임:

| 레이어 | 모듈 | 책임 | 의존 |
|---|---|---|---|
| Config | `config.py` | 하이퍼파라미터, 경로, 디바이스 결정, 로깅, 시드 헬퍼 | torch |
| Data | `data.py` | HF Hub 다운로드 → 필터링 → k-core → ID 매핑 → split → 그래프 빌드 → 캐시 | pandas, torch, PyG, huggingface_hub |
| Model | `model.py` | LightGCN 생성, 체크포인트 저장/로드 (메타데이터 포함) | PyG.LightGCN, torch |
| Train | `train.py` | BPR 학습 루프, 벡터화 음성 샘플링, LR scheduler, early stopping, 재개 | data, model, evaluate |
| Evaluate | `evaluate.py` | Full-ranking 평가 (배치 마스킹 + topk + HR/NDCG) | model, numpy |
| Service | `api/service.py` | 모듈 단위 싱글톤 상태(`_State`) — 추론 임베딩 + 추천 헬퍼 | data, model, torch |
| HTTP | `api/routes.py`, `api/main.py` | 요청 → 서비스 호출 → DTO, SPA fallback, OpenAPI | FastAPI, service |
| Frontend | `frontend/src/**` | UI/상태/네트워킹 | React, React Query, Tailwind, Vite |

---

## 3. 백엔드 아키텍처

### 3.1 계층 구조

```
            ┌──────────────────────────────────────────────────┐
            │                  HTTP layer                      │
            │  api/main.py  (FastAPI app, lifespan, static)    │
            │  api/routes.py (thin controllers)                │
            │  api/schemas.py (Pydantic DTO)                   │
            └──────────────────────┬───────────────────────────┘
                                   │ calls
            ┌──────────────────────▼───────────────────────────┐
            │                Service layer                     │
            │  api/service.py — module-level `_State`          │
            │   · is_loaded / load                             │
            │   · list_users / get_user / random_user          │
            │   · get_history / get_recommendations            │
            └──────────────────────┬───────────────────────────┘
                                   │ depends on
            ┌──────────────────────▼───────────────────────────┐
            │      Domain / Pipeline (offline + inference)     │
            │  data.preprocess  →  graph (PyG Data)            │
            │  model.create_model / load_checkpoint            │
            │  train.train (BPR loop)                          │
            │  evaluate.compute_metrics                        │
            └──────────────────────┬───────────────────────────┘
                                   │
            ┌──────────────────────▼───────────────────────────┐
            │                Infrastructure                    │
            │  torch / PyG / pandas / HF Hub / filesystem      │
            └──────────────────────────────────────────────────┘
```

**경계 규칙**

- 라우트는 비즈니스 로직을 가지지 않는다 → `service.*` 만 호출하고 DTO 로 직렬화.
- 서비스는 FastAPI 의존성을 임포트하지 않는다 → CLI/노트북에서도 재사용 가능.
- 도메인(model/train/evaluate/data)은 HTTP 를 모른다.
- 모든 모듈은 `config.cfg` 와 `config.logger` 만 공유 의존성으로 가진다.

### 3.2 핵심 데이터 객체

| 객체 | 형태 | 생성 위치 | 사용처 |
|---|---|---|---|
| `data` dict | `{graph, train_edges, val_edges, test_edges, num_users, num_items, train_history, asin_to_title, idx_to_item_asin, idx_to_user_id, user_map, item_map}` | `data.preprocess` | train, evaluate, service |
| `graph.edge_index` | `LongTensor[2, 2E]` undirected | `data.make_undirected` | LightGCN message passing |
| `train_history` | `dict[user_idx → set[item_idx_global]]` | `data.build_user_history` | 음성 샘플링 거절, 추천 마스킹 |
| Checkpoint | dict (state_dict + meta) | `model.save_checkpoint` | resume, 서빙 로딩 |
| `_State` | `(data, user_emb, item_emb, num_users, num_items)` | `service.load()` | 모든 inference 요청 |

### 3.3 추론 라이프사이클

1. **앱 기동**: `uvicorn api.main:app` →  `lifespan` 핸들러가 `service.load()` 호출.
2. **로드**: `data.preprocess()` (캐시 히트), `model.load_checkpoint()`,  CPU 로 `model.eval()` 전환.
3. **임베딩 precompute**: `model.get_embedding(edge_index)` 1 회만 수행 →  `[N, D]` 텐서를 `user_emb` / `item_emb` 로 슬라이스 후 contiguous 저장.
4. **요청 처리**: `get_recommendations(u, k)` 는 `user_emb[u] @ item_emb.T` 로 score 계산 → 학습 이력 마스킹 → `topk(k)`.
5. **결과**: 글로벌 인덱스 → ASIN → 제목 lookup 후 `RecommendationOut` DTO 반환.

성능 특성: 임베딩이 메모리에 상주하므로 요청당 비용은 `O(num_items)` 의 내적 1 회 + topk.

### 3.4 학습 라이프사이클

1. `seed_everything(cfg.seed)` — 결정론.
2. `preprocess()` — 캐시 미스면 다운로드/전처리.
3. **음성 샘플링 룩업 구축** (`_build_history_lookup`): CSR-style `(offsets, items)` 텐서로 패킹하여 GPU 에 한 번 전송.
4. 매 에폭:
   - 학습 엣지를 셔플 → 미니배치 단위로 잘라 BPR 손실 계산.
   - 한 배치당 메시지 패싱 1 회만 수행하여 임베딩 재사용 (`model.get_embedding`).
   - 벡터화된 거절 샘플링으로 음성 아이템 추출.
5. `eval_every` 마다 val 평가 → NDCG@10 기준 best 갱신 시 `best_model.pt` 저장, 매번 `last_model.pt` 도 저장.
6. early stopping 카운터가 `early_stop_patience // eval_every` 도달 시 종료.
7. 종료 후 best 체크포인트로 test 평가.

### 3.5 설정 (`config.py`)

`Config` 는 frozen 이 아닌 dataclass — 런타임에 `cfg.epochs` 등을 덮어쓸 수 있도록 의도된 설계.
`__post_init__` 에서 CUDA 가용 여부에 따라 `device` 결정, 필요한 디렉터리를 미리 생성.

| 분류 | 키 | 기본값 |
|---|---|---|
| 데이터 | `min_interactions`, `rating_threshold` | 3, 3.0 |
| 모델 | `embedding_dim`, `num_layers` | 64, 3 |
| 학습 | `lr`, `lambda_reg`, `batch_size`, `epochs`, `early_stop_patience`, `eval_every` | 1e-3, 1e-4, 4096, 200, 20, 5 |
| LR Scheduler | `lr_scheduler_factor`, `lr_scheduler_patience` | 0.5, 10 |
| 평가 | `eval_k` | [10, 20] |
| 데모 | `top_k_demo`, `demo_user_pool_size` | 10, 500 |
| 경로 | `data_dir`, `raw_dir`, `processed_dir`, `checkpoint_dir` | `backend/data/...` |

---

## 4. 프런트엔드 아키텍처

### 4.1 컴포넌트 트리

```
main.tsx
└─ QueryClientProvider
   └─ App
      ├─ Header (health 폴링: 30 s, useQuery)
      └─ Main / 3-column grid
         ├─ UserPicker         ← q 검색, listUsers(50, 0, q?)
         │   └─ Random 버튼     ← api.randomUser()
         ├─ Top-K slider       ← 로컬 state (1~50)
         ├─ HistoryList        ← getHistory(userIdx, 50)
         └─ RecommendationList ← getRecommendations(userIdx, k)
```

### 4.2 상태 관리

- **로컬 UI 상태**: `useState` — 선택된 사용자(`selected: UserOut | null`), Top-K 값(`k`), 검색어(`q`).
- **서버 상태 / 캐시**: `@tanstack/react-query`
  - `staleTime: 60_000` 으로 동일 입력에 대한 재요청 방지.
  - `enabled: userIdx !== null` 패턴으로 사용자 선택 전 호출 차단.
  - `health` 쿼리는 `refetchInterval: 30_000` 으로 라이브 핑.

### 4.3 네트워크 계층 (`api/client.ts`)

- 모든 호출은 `/api` prefix → Vite dev 프록시 / 프로덕션은 같은 origin.
- 한 곳에 모인 fetch 래퍼 `http<T>()` — 비-200 응답에서 JSON `detail` 추출하여 throw.
- DTO 타입은 이 파일에 인라인으로도, `types.ts` (OpenAPI 생성) 양쪽에서도 정의되어 있어 백엔드 스키마와 1:1 미러링.

### 4.4 스타일링

- Tailwind 3.4 유틸리티 클래스, 별도 디자인 시스템 없음.
- 디자인 토큰은 Tailwind 기본 팔레트 (slate/indigo/green/yellow/red).

---

## 5. 통신 / 단일 포트(single-port) 토폴로지

### 5.1 개발 환경

```
browser ──http──► Vite dev server (127.0.0.1:5173)
                      │
                      │ /api/*  (vite.config.ts proxy)
                      ▼
                  FastAPI (127.0.0.1:8000)
```

- 루트 `npm run dev` → `concurrently` 가 `dev:api` (`uvicorn --reload`) 와 `dev:web` (`vite`) 를 동시 기동.
- 브라우저 입장에서는 동일 origin 이라 CORS 가 필요 없다.

### 5.2 프로덕션 환경

```
browser ──http──► FastAPI (127.0.0.1:8000)
                      ├─ /api/*    → APIRouter
                      ├─ /assets/* → frontend/dist/assets (StaticFiles)
                      └─ /{path:*} → SPA fallback (index.html)
```

- `npm run build:web` 가 `frontend/dist` 생성.
- `api/main.py` 가 부팅 시 dist 폴더 존재를 감지하여 `StaticFiles` 와 SPA fallback 라우트를 등록.
- 라우트 우선순위: APIRouter 가 먼저 등록되어 `/api/*` 가 catch-all 보다 우선.
- `DISABLE_DOCS=1` 환경변수로 `/docs`, `/redoc`, `/openapi.json` 비활성화 가능.

### 5.3 REST 엔드포인트

| 메서드 | 경로 | 응답 | 비고 |
|---|---|---|---|
| GET | `/api/health` | `Health` | `model_loaded` 가 false 면 서비스 미준비 |
| GET | `/api/users?limit&offset&q` | `UserListOut` | 학습 이력 보유 사용자 풀에서 필터 |
| GET | `/api/users/random` | `UserOut` | 랜덤 1명 |
| GET | `/api/users/{user_idx}` | `UserOut` | 범위 밖이면 404 |
| GET | `/api/users/{user_idx}/history?limit` | `ItemOut[]` | 학습 데이터 이력 |
| GET | `/api/users/{user_idx}/recommendations?k` | `RecommendationOut[]` | 학습 이력 마스킹된 Top-K |

### 5.4 OpenAPI → TS 코드 생성

- 백엔드가 떠 있는 상태에서 `npm run typegen` 실행 → `frontend/src/api/types.ts` 갱신.
- 백엔드 스키마 변경 시 TS 컴파일 에러로 누락된 동기화를 즉시 발견.

---

## 6. 데이터 파이프라인

### 6.1 흐름

```
HF Hub                  Pandas filter chain               PyG
┌──────────────┐     ┌─────────────────────────┐     ┌──────────────┐
│ reviews.jsonl│──┐  │ filter_positive         │     │              │
│ meta.parquet │──┼─►│ kcore_filter (iter)     │────►│ Data(        │
└──────────────┘  │  │ build_mappings (offset) │     │  edge_index, │
                  │  │ leave_one_out_split     │     │  num_nodes   │
                  │  │ df_to_edge_index        │     │ )            │
                  │  │ make_undirected         │     │              │
                  │  └─────────────────────────┘     └──────────────┘
                  │              │                          │
                  │              ▼                          ▼
                  │      train_history dict          data/processed/data.pt
                  └──────────────────────────────────► (cache)
```

핵심 결정:

- **노드 공간 통합**: 사용자 인덱스 `[0, num_users)`, 아이템 인덱스 `[num_users, num_users + num_items)` 하나의 노드 공간에서 작동.
- **무방향 그래프**: LightGCN 의 양방향 message passing 을 위해 `make_undirected` 적용.
- **이력 셋 = 글로벌 인덱스**: `train_history[u]` 가 보유한 값은 이미 오프셋이 더해진 글로벌 인덱스. score 마스킹 시 `idx - num_users` 로 로컬 변환.
- **Leave-one-out + ≥3 이력**: 평가 가능한 사용자만 val/test 로 분리, 나머지는 모두 train 으로.
- **캐시**: `data/processed/data.pt` 가 존재하면 다운로드/전처리 전부 스킵.

### 6.2 학습 그래프 vs 서빙 그래프

학습 시 사용된 `train_edges`(unidirectional) 와 message passing 용 `graph.edge_index`(undirectional) 가 별개로 보존된다. 서빙은 message passing 그래프만 필요로 한다.

### 6.3 체크포인트

| 파일 | 시점 | 용도 |
|---|---|---|
| `best_model.pt` | NDCG@10 갱신 시 | 평가, 서빙 |
| `last_model.pt` | 평가 에폭마다 / early stop 시점 | `--resume` |

체크포인트 payload 에는 옵티마이저/스케줄러 상태, `best_ndcg`, `patience_counter`, 그래프 크기 등이 포함되어 재개 시 학습 곡선 손실이 거의 없도록 한다. 서빙 측에서는 `ckpt["num_nodes"] == num_users + num_items` 를 검증해 데이터/모델 불일치를 조기 차단.

---

## 7. 빌드 / 실행 / 배포

### 7.1 명령 매트릭스

| 시나리오 | 명령 | 비고 |
|---|---|---|
| 로컬 dev (전체) | `npm run dev` (루트) | concurrently 로 FastAPI + Vite 동시 기동 |
| dev API only | `npm run dev:api` | `uvicorn --reload` |
| dev web only | `npm run dev:web` | Vite 5173 |
| 데이터 전처리만 | `cd backend && python run.py --skip-train` | 캐시가 없을 때 데이터셋 다운로드 |
| 전체 파이프라인 | `python run.py` | data → train → 종료 |
| 학습 재개 | `python run.py --resume` | `last_model.pt` 에서 |
| 프런트 프로덕션 빌드 | `npm run build:web` | dist 생성 |
| 단일 포트 프로덕션 서빙 | `npm run serve` | 빌드 dist 를 FastAPI 가 서빙 |
| OpenAPI 동기화 | `npm run typegen` | API ↔ TS 타입 동기 |
| (LEGACY) Gradio | `python run.py --legacy-gradio` | 신규 기능 금지 |

### 7.2 런타임 요구사항

- Python 3.10 (예: conda env `py310_pt`) + PyTorch + torch-geometric. CUDA 는 학습용 권장, 서빙은 CPU 로 동작.
- Node 18.x (`.nvmrc` = 18.20.8) + npm ≥ 9.
- 디스크: HF 다운로드 캐시 + `backend/data/processed/data.pt` + 체크포인트 ≈ 수십 MB 수준.

### 7.3 가능한 향후 토폴로지 확장

본 MVP 는 단일 인스턴스 단일 사용자(데모)를 가정한다. 멀티 사용자/세션, 인증, 데이터 갱신 워크플로, 캐시 무효화 등은 현재 범위 밖이다. 확장 시 자연스러운 진입점:

- 서비스 레이어를 stateless POJO 로 분리 → 다중 워커.
- 임베딩 store 를 외부 캐시(Redis 등)로 이동.
- 학습을 Celery/Airflow 등 외부 잡 러너로 분리.

---

## 8. 비기능 요건 / 설계 트레이드오프

| 항목 | 결정 | 이유 |
|---|---|---|
| 서빙 디바이스 | CPU 고정 | 단일 사용자 MVP — 예측 가능성 + GPU 메모리 절약 |
| 핸들러 시그니처 | sync `def` | PyTorch inference 가 blocking → FastAPI 가 threadpool 에 디스패치하여 event loop 유지 |
| 상태 보관 | 모듈 전역 `_state` | 싱글톤 의존성 주입 제거, 부팅 시 lifespan 으로 초기화 |
| 음성 샘플링 | 벡터화 거절 샘플링 (CSR 룩업) | Python 루프 제거, GPU 친화적 |
| 평가 | Full-ranking (모든 아이템) | 작은 카탈로그(<2k) 이므로 sampled-eval 의 노이즈를 회피 |
| 데이터 캐시 | `data.pt` 단일 pickle | 빠른 재기동, 데이터 변경 시 명시적 삭제 필요 |
| API ↔ UI 계약 | snake_case 1:1 미러링 + 코드 생성 | 두 언어 사이의 drift 방지 |
| Gradio 데모 | LEGACY 마킹 | FastAPI + React 로 일원화, 코드 동결 |

---

## 9. 변경 가이드라인

- **새 엔드포인트 추가**: `service` 에 함수 추가 → `schemas` 에 DTO → `routes` 에 라우트 → `npm run typegen` → 프런트에서 사용.
- **새 하이퍼파라미터**: `config.Config` 에 필드 추가 후 `run.py` CLI 옵션이나 `train.py` 인자 경유로 노출.
- **모델 교체**: `model.create_model` 시그니처 유지 + `LightGCN.get_embedding`/`recommendation_loss` 와 동등한 인터페이스를 만족하면 train/service 코드 수정 최소화.
- **데이터셋 교체**: `data.download_reviews`/`download_metadata` 만 변경하고 컬럼 (`user`, `item`, `rating`, `timestamp`) 계약 유지.
- **Gradio 코드**: 수정 금지 (legacy 동결).
