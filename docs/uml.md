# UML Diagrams

> GNN Product Recommender — UML 다이어그램 모음

본 문서의 모든 다이어그램은 [Mermaid](https://mermaid.js.org/) 문법으로 작성되었습니다. GitHub, VS Code(Mermaid 확장), IntelliJ 등에서 바로 렌더링됩니다.
시스템 개요는 [`architecture.md`](./architecture.md) 를 참고하세요.

목차:
1. [컴포넌트 다이어그램](#1-컴포넌트-다이어그램)
2. [패키지 다이어그램](#2-패키지-다이어그램)
3. [백엔드 클래스 다이어그램](#3-백엔드-클래스-다이어그램)
4. [프런트엔드 클래스/컴포넌트 다이어그램](#4-프런트엔드-클래스컴포넌트-다이어그램)
5. [DTO / 데이터 모델 다이어그램](#5-dto--데이터-모델-다이어그램)
6. [시퀀스 다이어그램 — 추천 요청](#6-시퀀스-다이어그램--추천-요청)
7. [시퀀스 다이어그램 — 서버 부팅 / 모델 로드](#7-시퀀스-다이어그램--서버-부팅--모델-로드)
8. [시퀀스 다이어그램 — 학습 파이프라인](#8-시퀀스-다이어그램--학습-파이프라인)
9. [시퀀스 다이어그램 — 평가 루프](#9-시퀀스-다이어그램--평가-루프)
10. [활동 다이어그램 — 데이터 전처리](#10-활동-다이어그램--데이터-전처리)
11. [상태 다이어그램 — 학습 사이클](#11-상태-다이어그램--학습-사이클)
12. [배포 다이어그램](#12-배포-다이어그램)

---

## 1. 컴포넌트 다이어그램

런타임 컴포넌트와 의존성. 단일 포트(single-port) 토폴로지를 보여줍니다.

```mermaid
graph LR
    Browser["🌐 Browser<br/>(SPA)"]

    subgraph Frontend["frontend (Vite + React)"]
        App["App.tsx"]
        UP["UserPicker"]
        HL["HistoryList"]
        RL["RecommendationList"]
        ApiClient["api/client.ts<br/>(fetch wrapper)"]
        RQ["React Query<br/>cache"]
    end

    subgraph Vite["Vite dev server :5173"]
        Proxy["/api proxy"]
    end

    subgraph Backend["backend (FastAPI :8000)"]
        Main["api/main.py<br/>app + lifespan"]
        Routes["api/routes.py"]
        Service["api/service.py<br/>(_State singleton)"]
        Schemas["api/schemas.py"]
        Static["StaticFiles<br/>+ SPA fallback"]
    end

    subgraph Domain["Domain modules"]
        Data["data.py"]
        Model["model.py"]
        Train["train.py"]
        Eval["evaluate.py"]
        Cfg["config.py"]
    end

    subgraph Infra["Infrastructure"]
        Torch["PyTorch + PyG"]
        HF["HuggingFace Hub"]
        FS["Filesystem<br/>(data/, checkpoints/)"]
    end

    Browser -->|HTML/JS| App
    App --> UP & HL & RL
    UP & HL & RL --> RQ --> ApiClient
    ApiClient -->|HTTP /api/*| Proxy
    Proxy -->|proxy /api| Routes

    Browser -. prod-only .-> Static

    Main --> Routes
    Main -->|lifespan startup| Service
    Routes --> Service
    Routes --> Schemas
    Service --> Data
    Service --> Model
    Train --> Data
    Train --> Model
    Train --> Eval

    Data --> HF
    Data --> FS
    Model --> FS
    Service --> Torch
    Train --> Torch
    Eval --> Torch
    Cfg -. used by all .-> Service
    Cfg -. used by all .-> Train
    Cfg -. used by all .-> Data
    Cfg -. used by all .-> Eval
```

---

## 2. 패키지 다이어그램

코드 베이스의 모듈/패키지 의존 그래프(점선은 동적/lazy import).

```mermaid
graph TD
    subgraph backend
        direction TB
        config["config"]
        data["data"]
        model["model"]
        evaluate["evaluate"]
        train["train"]
        run["run"]
        gradio["app_gradio<br/>(LEGACY)"]
        subgraph api
            api_main["api.main"]
            api_routes["api.routes"]
            api_schemas["api.schemas"]
            api_service["api.service"]
        end
    end

    data --> config
    model --> config
    evaluate --> config
    train --> config
    train --> data
    train --> model
    train --> evaluate
    run -.-> data
    run -.-> train
    run -.-> gradio
    api_main --> api_routes
    api_main --> api_service
    api_routes --> api_service
    api_routes --> api_schemas
    api_service --> config
    api_service --> data
    api_service --> model
    gradio --> config
    gradio --> data
    gradio --> model

    subgraph frontend
        direction TB
        fmain["main.tsx"]
        fapp["App.tsx"]
        fpicker["components/UserPicker"]
        fhist["components/HistoryList"]
        frec["components/RecommendationList"]
        fclient["api/client.ts"]
        ftypes["api/types.ts<br/>(generated)"]
    end

    fmain --> fapp
    fapp --> fpicker & fhist & frec
    fpicker --> fclient
    fhist --> fclient
    frec --> fclient
    fapp --> fclient
```

---

## 3. 백엔드 클래스 다이어그램

핵심 클래스/데이터클래스/모듈 단위 함수 그룹.

```mermaid
classDiagram
    class Config {
        +str hf_dataset
        +int min_interactions
        +float rating_threshold
        +int embedding_dim
        +int num_layers
        +float lr
        +float lambda_reg
        +int batch_size
        +int epochs
        +int early_stop_patience
        +int eval_every
        +int seed
        +int lr_scheduler_patience
        +float lr_scheduler_factor
        +list[int] eval_k
        +int top_k_demo
        +Path data_dir
        +Path raw_dir
        +Path processed_dir
        +Path checkpoint_dir
        +torch.device device
        +__post_init__()
    }

    class LightGCN {
        +int num_nodes
        +int embedding_dim
        +int num_layers
        +get_embedding(edge_index) Tensor
        +recommendation_loss(pos, neg, node_id, lambda_reg) Tensor
        +forward(edge_index, edge_label_index) Tensor
    }

    class ModelModule {
        <<module: model.py>>
        +create_model(num_nodes, dim?, layers?) LightGCN
        +save_checkpoint(model, optimizer, epoch, metrics, ...)
        +load_checkpoint(path?) (LightGCN, dict)
    }

    class DataModule {
        <<module: data.py>>
        +download_reviews() DataFrame
        +download_metadata() dict
        +filter_positive(df) DataFrame
        +kcore_filter(df, min_k) DataFrame
        +build_mappings(df) tuple
        +leave_one_out_split(df) tuple
        +df_to_edge_index(df, umap, imap) Tensor
        +make_undirected(edge_index) Tensor
        +build_user_history(edge_index, num_users) dict
        +preprocess() dict
    }

    class TrainModule {
        <<module: train.py>>
        +_build_history_lookup(history, n_u, n_n, device) tuple
        +_sample_negatives(pos_src, n_u, n_n, off, items, gen) Tensor
        +train_epoch(model, opt, edge_index, train_edges, ...)
        +train(epochs_override?, resume?) tuple
    }

    class EvaluateModule {
        <<module: evaluate.py>>
        +compute_metrics(model, edge_index, eval_edges, history, n_u, n_i, k_list?, batch_size?) dict
    }

    class _State {
        +dict data
        +Tensor user_emb
        +Tensor item_emb
        +int num_users
        +int num_items
    }

    class ServiceModule {
        <<module: api.service>>
        -_State _state
        +load()
        +is_loaded() bool
        +get_counts() tuple
        +list_users(limit, offset, q?) tuple
        +get_user(user_idx) dict
        +random_user() dict
        +get_history(user_idx, limit) list
        +get_recommendations(user_idx, k) list
    }

    class FastAPIApp {
        <<api.main>>
        +FastAPI app
        +lifespan(app)
        +spa_fallback(full_path, request)
    }

    class RoutesModule {
        <<module: api.routes>>
        +APIRouter router
        +health()
        +list_users(limit, offset, q?)
        +random_user()
        +get_user(user_idx)
        +get_history(user_idx, limit)
        +get_recommendations(user_idx, k)
    }

    class Health
    class UserOut
    class ItemOut
    class RecommendationOut
    class UserListOut
    class Error
    Health : +str status
    Health : +bool model_loaded
    Health : +int num_users
    Health : +int num_items
    UserOut : +int user_idx
    UserOut : +str original_id
    UserOut : +int history_size
    ItemOut : +int item_idx
    ItemOut : +str asin
    ItemOut : +str title
    RecommendationOut : +int item_idx
    RecommendationOut : +str asin
    RecommendationOut : +str title
    RecommendationOut : +float score
    UserListOut : +int total
    UserListOut : +list[UserOut] items
    Error : +str detail
    Error : +str code

    ModelModule ..> LightGCN : creates
    DataModule ..> Config : uses
    ServiceModule --> _State : owns
    ServiceModule ..> DataModule : preprocess()
    ServiceModule ..> ModelModule : load_checkpoint()
    ServiceModule ..> LightGCN : get_embedding()
    TrainModule ..> DataModule
    TrainModule ..> ModelModule
    TrainModule ..> EvaluateModule
    TrainModule ..> LightGCN
    EvaluateModule ..> LightGCN
    FastAPIApp ..> RoutesModule : include_router
    FastAPIApp ..> ServiceModule : lifespan→load()
    RoutesModule ..> ServiceModule
    RoutesModule ..> Health
    RoutesModule ..> UserOut
    RoutesModule ..> ItemOut
    RoutesModule ..> RecommendationOut
    RoutesModule ..> UserListOut
```

---

## 4. 프런트엔드 클래스/컴포넌트 다이어그램

React 컴포넌트 props, 외부 의존, API 메서드.

```mermaid
classDiagram
    class Main {
        <<entry: main.tsx>>
        +QueryClient queryClient
        +render()
    }

    class App {
        +UserOut|null selected
        +number k
        -useQuery~Health~ health
        +render()
    }

    class UserPicker {
        +UserOut|null selected
        +function onSelect
        -string q
        -useQuery~UserListOut~ data
        +onRandom()
    }

    class HistoryList {
        +number|null userIdx
        -useQuery~ItemOut[]~ data
    }

    class RecommendationList {
        +number|null userIdx
        +number k
        -useQuery~RecommendationOut[]~ data
    }

    class ApiClient {
        <<module: api/client.ts>>
        -string BASE
        -http~T~(path, init?) Promise~T~
        +health() Promise~Health~
        +listUsers(limit, offset, q?) Promise~UserListOut~
        +randomUser() Promise~UserOut~
        +getUser(idx) Promise~UserOut~
        +getHistory(idx, limit?) Promise~ItemOut[]~
        +getRecommendations(idx, k?) Promise~RecommendationOut[]~
    }

    class UserOut {
        +number user_idx
        +string original_id
        +number history_size
    }
    class ItemOut {
        +number item_idx
        +string asin
        +string title
    }
    class RecommendationOut {
        +number item_idx
        +string asin
        +string title
        +number score
    }
    class UserListOut {
        +number total
        +UserOut[] items
    }
    class Health {
        +string status
        +boolean model_loaded
        +number num_users
        +number num_items
    }

    Main --> App
    App --> UserPicker
    App --> HistoryList
    App --> RecommendationList
    App ..> ApiClient
    UserPicker ..> ApiClient
    HistoryList ..> ApiClient
    RecommendationList ..> ApiClient
    ApiClient ..> Health
    ApiClient ..> UserOut
    ApiClient ..> UserListOut
    ApiClient ..> ItemOut
    ApiClient ..> RecommendationOut
```

---

## 5. DTO / 데이터 모델 다이어그램

백엔드 Pydantic 모델과 프런트엔드 TS 인터페이스가 1:1 미러링됨을 보여줍니다.

```mermaid
classDiagram
    direction LR

    class Health {
        +str status
        +bool model_loaded
        +int num_users
        +int num_items
    }
    class UserOut {
        +int user_idx
        +str original_id
        +int history_size
    }
    class ItemOut {
        +int item_idx
        +str asin
        +str title
    }
    class RecommendationOut {
        +int item_idx
        +str asin
        +str title
        +float score
    }
    class UserListOut {
        +int total
        +list[UserOut] items
    }
    UserListOut o-- UserOut : items
    RecommendationOut --|> ItemOut : extends (score 추가)

    class PreprocessedData {
        <<dict: data.preprocess()>>
        +Data graph
        +Tensor train_edges
        +Tensor val_edges
        +Tensor test_edges
        +int num_users
        +int num_items
        +dict[int, set[int]] train_history
        +dict[str, str] asin_to_title
        +dict[int, str] idx_to_item_asin
        +dict[int, str] idx_to_user_id
        +dict[str, int] user_map
        +dict[str, int] item_map
    }

    class Checkpoint {
        <<dict>>
        +dict model_state_dict
        +dict optimizer_state_dict
        +int epoch
        +dict metrics
        +int num_nodes
        +int embedding_dim
        +int num_layers
        +dict? scheduler_state_dict
        +float? best_ndcg
        +int? patience_counter
    }

    class _State {
        +PreprocessedData data
        +Tensor user_emb
        +Tensor item_emb
        +int num_users
        +int num_items
    }

    _State *-- PreprocessedData
```

---

## 6. 시퀀스 다이어그램 — 추천 요청

사용자가 UserPicker 에서 한 명을 선택하면 발생하는 전체 호출 경로.

```mermaid
sequenceDiagram
    actor U as User
    participant B as Browser
    participant V as Vite Proxy (5173)
    participant R as FastAPI Routes
    participant S as service.py
    participant T as torch / PyG

    U->>B: Click user in UserPicker
    B->>B: setSelected(user)
    par history
        B->>V: GET /api/users/{idx}/history?limit=50
        V->>R: forward
        R->>S: get_history(idx, 50)
        S-->>R: list[ItemOut dict]
        R-->>V: 200 + JSON
        V-->>B: JSON
    and recommendations
        B->>V: GET /api/users/{idx}/recommendations?k=10
        V->>R: forward
        R->>S: get_recommendations(idx, k)
        S->>T: user_emb[idx] @ item_emb.T
        T-->>S: scores [num_items]
        S->>S: mask train_history
        S->>T: scores.topk(k)
        T-->>S: top scores + indices
        S-->>R: list[RecommendationOut dict]
        R-->>V: 200 + JSON
        V-->>B: JSON
    end
    B->>U: Render HistoryList + RecommendationList
```

---

## 7. 시퀀스 다이어그램 — 서버 부팅 / 모델 로드

`uvicorn api.main:app` 부팅부터 첫 요청 준비까지.

```mermaid
sequenceDiagram
    participant U as uvicorn
    participant A as api.main (FastAPI)
    participant Sv as api.service
    participant D as data.preprocess
    participant M as model.load_checkpoint
    participant FS as Filesystem
    participant T as torch / PyG

    U->>A: import app
    A->>A: build FastAPI(lifespan=lifespan)
    A->>A: include_router(api_router)
    A->>A: mount /assets (if dist exists)
    U->>A: lifespan startup
    A->>Sv: service.load()
    Sv->>D: preprocess()
    D->>FS: read data/processed/data.pt (cache)
    FS-->>D: data dict
    D-->>Sv: data
    Sv->>M: load_checkpoint()
    M->>FS: read checkpoints/best_model.pt
    FS-->>M: state_dict + meta
    M-->>Sv: (model, ckpt)
    Sv->>Sv: validate num_nodes
    Sv->>T: model.eval(), .to("cpu")
    Sv->>T: model.get_embedding(edge_index)
    T-->>Sv: emb [N, D]
    Sv->>Sv: split into user_emb / item_emb, store _State
    Sv-->>A: ready
    A-->>U: ready to serve
```

---

## 8. 시퀀스 다이어그램 — 학습 파이프라인

`python run.py` 가 부르는 전 과정.

```mermaid
sequenceDiagram
    actor Dev as Developer
    participant Run as run.py
    participant Cfg as config
    participant D as data.preprocess
    participant Tr as train.train
    participant Mdl as model
    participant Ev as evaluate
    participant FS as Filesystem

    Dev->>Run: python run.py [--resume] [--epochs N]
    Run->>Cfg: seed_everything(seed)
    Run->>D: preprocess()
    alt cache miss
        D->>D: download_reviews / download_metadata (HF Hub)
        D->>D: filter_positive → kcore → mappings → split
        D->>D: df_to_edge_index → make_undirected → build_user_history
        D->>FS: torch.save(data.pt)
    else cache hit
        D->>FS: load data.pt
    end
    D-->>Run: data
    Run->>Tr: train(epochs_override, resume)
    Tr->>Tr: _build_history_lookup → CSR (offsets, items)
    alt resume
        Tr->>FS: load last_model.pt
        Tr->>Mdl: create_model(num_nodes, dim, layers)
    else fresh
        Tr->>Mdl: create_model(num_nodes)
    end
    Tr->>Mdl: optimizer + ReduceLROnPlateau
    loop epoch = start..epochs
        Tr->>Tr: train_epoch (BPR loop)
        loop minibatch
            Tr->>Tr: _sample_negatives (vectorized rejection)
            Tr->>Mdl: get_embedding(edge_index)
            Tr->>Tr: compute BPR loss
            Tr->>Mdl: optimizer.step()
        end
        opt eval_every epochs
            Tr->>Ev: compute_metrics(val)
            Ev-->>Tr: HR@K, NDCG@K
            Tr->>Tr: scheduler.step(ndcg10)
            alt ndcg improved
                Tr->>FS: save best_model.pt
            else no improvement
                Tr->>Tr: patience_counter++
                opt patience exhausted
                    Tr->>FS: save last_model.pt
                    Tr->>Tr: break
                end
            end
            Tr->>FS: save last_model.pt
        end
    end
    Tr->>Mdl: load_checkpoint() (best)
    Tr->>Ev: compute_metrics(test)
    Ev-->>Tr: test metrics
    Tr-->>Run: (best_model, test_metrics)
    Run-->>Dev: done
```

---

## 9. 시퀀스 다이어그램 — 평가 루프

`evaluate.compute_metrics` 의 내부 흐름.

```mermaid
sequenceDiagram
    participant Caller as caller (train / cli)
    participant E as compute_metrics
    participant M as LightGCN
    participant T as torch

    Caller->>E: compute_metrics(model, edge_index, eval_edges, history, n_u, n_i)
    E->>E: build ground_truth dict (user → set[item])
    E->>M: eval mode
    E->>M: get_embedding(edge_index)
    M-->>E: emb [N, D]
    E->>E: split into user_emb / item_emb
    loop each batch of eval users
        E->>T: scores = user_emb[batch] @ item_emb.T
        E->>E: mask training items (set to -inf)
        E->>T: scores.topk(max_k)
        loop each user in batch
            E->>E: hit check (any topk in gt?)
            E->>E: NDCG = DCG / IdealDCG
        end
    end
    E-->>Caller: dict{hr@k, ndcg@k}
```

---

## 10. 활동 다이어그램 — 데이터 전처리

```mermaid
flowchart TD
    Start([preprocess called]) --> Cache{cache exists?}
    Cache -- yes --> Load[torch.load data.pt] --> End([return data])
    Cache -- no --> DL1[download_reviews]
    DL1 --> DL2[download_metadata]
    DL2 --> Pos[filter_positive: rating ≥ 3.0]
    Pos --> Kcore[kcore_filter min_k iter convergence]
    Kcore --> Few{< 10k & min_k > 3?}
    Few -- yes --> Pos2[re-run with 3-core] --> Map
    Few -- no --> Map[build_mappings, item_idx += num_users]
    Map --> Split[leave_one_out_split]
    Split --> Edges[df_to_edge_index, make_undirected]
    Edges --> Hist[build_user_history]
    Hist --> Save[torch.save data.pt] --> End
```

---

## 11. 상태 다이어그램 — 학습 사이클

```mermaid
stateDiagram-v2
    [*] --> Initializing
    Initializing --> ResumeOrFresh
    state ResumeOrFresh <<choice>>
    ResumeOrFresh --> Resuming: --resume & last_model.pt
    ResumeOrFresh --> FreshInit: otherwise
    Resuming --> Training
    FreshInit --> Training

    state Training {
        [*] --> BatchLoop
        BatchLoop --> EvalGate: epoch complete
        EvalGate --> EvalCycle: epoch % eval_every == 0
        EvalGate --> NextEpoch: otherwise
        EvalCycle --> BestSaved: NDCG improved
        EvalCycle --> PatienceInc: no improvement
        BestSaved --> SaveLast
        PatienceInc --> EarlyStopCheck
        EarlyStopCheck --> SaveLast: not exhausted
        EarlyStopCheck --> EarlyStop: exhausted
        SaveLast --> NextEpoch
        NextEpoch --> BatchLoop: epoch < epochs
        NextEpoch --> Finished: epoch == epochs
        EarlyStop --> Finished
    }
    Training --> TestEvaluating: Finished
    TestEvaluating --> [*]
```

---

## 12. 배포 다이어그램

개발 / 프로덕션 토폴로지를 한 장으로 비교.

```mermaid
graph TB
    subgraph Dev["Development (npm run dev)"]
        direction LR
        DBrowser["Browser"]
        DVite["Vite (5173)<br/>SPA + /api proxy"]
        DApi["uvicorn --reload<br/>FastAPI (8000)"]
        DDist[("frontend/<br/>src/*<br/>(HMR)")]
        DCkpt[("backend/<br/>data/checkpoints/")]
        DBrowser -- "HTTP" --> DVite
        DVite -- "/api/*" --> DApi
        DVite -. dev assets .- DDist
        DApi -. read .- DCkpt
    end

    subgraph Prod["Production (npm run serve)"]
        direction LR
        PBrowser["Browser"]
        PApi["uvicorn<br/>FastAPI (8000)<br/>+ StaticFiles + SPA fallback"]
        PDist[("frontend/dist/<br/>(npm run build:web)")]
        PCkpt[("backend/<br/>data/checkpoints/")]
        PProcessed[("backend/<br/>data/processed/data.pt")]
        PBrowser -- "HTTP /, /api/*" --> PApi
        PApi -. serve .- PDist
        PApi -. lifespan load .- PCkpt
        PApi -. lifespan load .- PProcessed
    end
```

---

## 부록 — Mermaid 렌더링 팁

- **GitHub**: 자동 렌더링 (라이트/다크 모두).
- **VS Code**: `Markdown Preview Mermaid Support` 확장 설치.
- **JetBrains**: Markdown plugin 의 "Mermaid" 옵션을 활성화.
- **로컬 PNG/SVG 추출**: `npx @mermaid-js/mermaid-cli@latest -i uml.md -o out.svg` (코드블록 단위로 가능).
