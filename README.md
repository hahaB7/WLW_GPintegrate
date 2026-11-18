# WLW_GPintegrate

WLW_GPintegrate 是一套用來串接病理報告、醫學文獻與全玻片影像（WSI）的研究型推論框架。
它把五位開發者各自負責的模型整理成一條可重現的批次流程：從病理報告找出診斷項目、檢索相關
文獻、在 WSI 上提出候選 ROI、產生視覺判讀條件，再以影像模型配對與篩選證據。

這個專案採用 contract-first 設計。元件之間不直接 import 對方的研究程式，而是透過版本化 JSON
artifact 交換資料。每個元件都有自己的 Docker image、設定檔與外部資產，整體流程則由
`pipeline/run_pipeline.py` 負責編排。因此，單一元件可以獨立開發與測試，整合時也能清楚追蹤每一步
用了什麼模型、設定與輸入。

2026-10-29 已在 RTX 4090、正式 Docker images、三張真實 MRXS 與完整 `reference/` 外部資產上跑通
production pipeline。每次正式執行的案例、映像、設定與時間會寫入 sibling
`run/output/pipeline/run_manifest.json`，完整過程則保存在 `run/logs/`。

## 快速開始

### 1. 準備目錄

程式庫、外部資產與執行資料必須位於同一層：

```text
lab_23/
├── WLW_GPintegrate/                 # 本程式庫
├── reference/                       # 模型、checkpoint、corpus、template；執行時唯讀
└── run/                             # 輸入、輸出、cache 與 log
    ├── input/pipeline/
    ├── output/
    ├── cache/
    └── logs/
```

正式輸入至少包含：

```text
run/input/pipeline/
├── cases.json                       # 多案例 CaseListInput
├── A_literature.json                # 可由 prepare_literature 產生
└── *.mrxs / *.svs / ...             # cases.json 所引用的 WSI
```

各元件需要的權重與參照檔放在 `reference/person_a/` 至 `reference/person_e/`。完整檔名、版本、
SHA-256 與授權資訊記錄在各元件的 `external-assets.yaml`；這些大型或受控資產不會被包進 Git 或
Docker image。

### 2. 建立正式文獻 artifact

在專案根目錄執行：

```bash
python pipeline/run_prepare_literature.py
```

production profile 會由 `reference/person_b/rag_database` 建立
`run/input/pipeline/A_literature.json`。若檔案已準備完成且通過 schema，也可以直接進入下一步。

### 3. 執行完整 production pipeline

```bash
python pipeline/run_pipeline.py
```

無參數時預設為 `production`。Runner 會使用 Docker、唯讀掛載 `/reference`、可寫掛載 `/run`，
逐案執行六個 stage。成功後可在以下位置查看結果：

```text
run/output/pipeline/
├── run_manifest.json
└── cases/
    └── <case_id>/
        ├── case_input.json
        ├── D_dx_pairs.json
        ├── E_rois.json
        ├── F_chunks.json
        ├── G_queries.json
        ├── H_matches.json
        └── I_selected_rois.json
```

完整 stdout/stderr 會另存於 `run/logs/pipeline-production-<timestamp>.log`。`run_manifest.json` 記錄
完成案例、每個 stage 使用的 config、Docker image 與執行時間，是確認一次正式 run 是否完整的第一個入口。

只執行指定案例時，可重複使用 `--case-id`：

```bash
python pipeline/run_pipeline.py --case-id S1202317
```

## 框架如何運作

整體流程同時處理「文字證據」與「影像證據」，最後在 H、I 階段匯合：

```text
CaseList ─> Person A／報告解析 ─> [D] ─> Person B／文獻檢索 ─> [F] ─> Person A／視覺條件 ─> [G] ─┐
                                                                                                  ├─> Person D／Matching ─> [H] ─┐
CaseList ─> Person C／WSI ROI 提案 ─────────────────────────────────────────────────────> [E] ─┘                                │
                                                                                                                                    ├─> Person E／CLEE ─> [I]
[D] ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

更精確地說：

1. Person A 將原始病理報告拆成結構化診斷項目 D。
2. Person B 依 D 從正式文獻庫檢索相關段落，產生 F。
3. Person A 結合 D 與 F，以 learnable soft prompt 產生視覺屬性條件 G。
4. Person C 獨立讀取 WSI，以 CONCH 特徵與 vMF 分群提出候選 ROI，產生 E。
5. Person D 對 E 的 ROI 執行 PLIP + CONCH 視覺屬性抽取，並依 G 做 matching，產生 H。
6. Person E 結合 D 與 H，以 CLEE 決定最後的證據 ROI，產生 I。

`pipeline/run_pipeline.py` 會把多案例輸入拆成單案 artifact；每個 D–I 檔案只包含一個 `case_id`，
避免不同病人的資料在元件邊界混合。`integration/compose.yaml` 則保留單案例的靜態容器拓撲，適合檢查
部署設定與服務間掛載。

## 五個元件

### Person A：報告解析與視覺條件生成

版本：`0.10.0`

Person A 有兩個入口：

- `components.person_a.report_decompose`：將 CaseList 中的病理報告轉成 `D.DxPairs`。
- `components.person_a.query_generation`：結合 D 與 Person B 的 F，產生 `G.VisualAttributeQueries`。

正式 Query Generation 使用固定 revision 的 Gemma 3 1B 與 64-token Stage 2 soft prompt，模型完整放在
單張 GPU，不使用 CPU offload。模型輸出必須是嚴格 JSON；runtime 會將結果投影到 condition-free
canonical template：多出的節點丟棄，缺少、型別錯誤或 condition 非法的 canonical option 補為
`Not_Mentioned`，並把所有正規化明細寫入 G provenance。若原始診斷文字產生無法解析的輸出，
production profile 可改用已驗證的 canonical diagnosis，以同一模型、soft prompt 與 chunks 重試；
兩次嘗試的狀態與 response hash 都會寫入 provenance，不會退回 deterministic reference backend。

報告解析採 hospital routing：目前 VGHTC 由正式 regex extractor 處理；其他院別與 MedGemma fallback
規則請見 [Person A README](components/person_a/README.md)。

### Person B：文獻準備與檢索

版本：`0.5.0`

Person B 負責兩件事：

- `components.person_b.prepare_literature`：將正式 corpus 正規化成 `A.Literature`。
- `components.person_b.knowledge_retrieval`：依 D 的診斷項目進行 dense retrieval，產生 `F.Chunks`。

目前 production 使用 `reference/person_b/rag_database`、固定的 knowledge base 與 MedEmbed embedding
model；每個有支援的診斷項目取 top-3 chunks，交給 Person A 的 soft-prompt 模型。

詳細資產與檢索設定請見 [Person B README](components/person_b/README.md)。

### Person C：WSI Interest Pattern 與 ROI 提案

版本：`0.1.0`

Person C 的 `components.person_c.interest_pattern` 直接讀取 WSI，使用 CONCH dense features、BCSS
vocabulary、vMF clustering 與 zero-shot 命名提出 ROI，輸出 `E.ROIs`。ROI JSON 只保存座標、MPP 與
metadata，不把影像像素塞入 artifact；Person D、E 會依 WSI 路徑與 `level0_info` 重建影像區域。

大型 WSI 的 accumulator 會依可用 VRAM 自動選擇策略：可容納時使用單次 `joint_gpu`；兩個 feature
accumulator 無法同時放入 GPU 時，使用數學等價的 `two_pass_gpu`，將第一路正規化結果移到 system RAM
後再處理第二路。正式 RTX 4090 執行曾處理超過 517 萬個 compact cells。

座標規則、硬體需求與演算法限制請見 [Person C README](components/person_c/README.md)。

### Person D：視覺屬性抽取與 Matching

版本：`0.2.0`

Person D 的 `components.person_d.visual_filter` 讀取 E 與 G，使用正式 PLIP、CONCH、視覺 prompt 與
few-shot prototypes 對每個 ROI 產生視覺屬性分數，再依 G 的 Must／Possibly condition 執行 matching，
輸出 `H.MatchedROIs`。

production 固定使用 `components/person_d/configs/native.json`，且 `matching.enable=true`。
`native.example.json` 的 matching-disabled passthrough 只供受控 fixture 使用，不會被無參數 production
runner 選中。H 的 `reference_versions` 會記錄模型是否載入、資產 hash、matching policy 與 few-shot
layout，便於區分「模型已執行但沒有 ROI 通過」和「matching 根本沒有開啟」。

詳細 mapping 規則請見 [Person D README](components/person_d/README.md)。

### Person E：CLEE 證據選擇

版本：`1.1.0`

Person E 的 `components.person_e.clee` 結合 D 與 H，以正式 CLEE checkpoint、threshold bundle 與
label space 選擇最終證據 ROI，輸出 `I.CLEESelectedROIs`。

CLEE 只對通過 Person D 且屬於有效診斷空間的 ROI 執行 forward。若沒有合格 ROI，native backend
會明確記錄 `skipped_no_eligible_rois`，而不是偷偷放寬 D 的 matching 或產生假結果。因此，執行成功
不代表每個案例都一定會得到 selected ROI；判讀 I 時應同時查看 `execution_statuses`、計數與每個 ROI
的 `selection_history`。

詳細 threshold 與 fallback policy 請見 [Person E README](components/person_e/README.md)。

## Artifact 與 Contract

所有元件交換的 JSON 都由 `contracts/schemas/` 驗證。Runner 在讀取輸入與寫出結果時都會檢查
contract 名稱、schema version、必要欄位及多輸入的 `case_id` 一致性。

| 代號 | Contract | 用途 |
|---|---|---|
| A | `A.Literature@2.0` | 正規化文獻與 sections |
| B | `B.ReportTables@1.0` | 上游報告表格 manifest；目前主 pipeline 從 CaseList 開始 |
| C | 無獨立 canonical artifact | 保留代號，尚未定義交換格式 |
| D | `D.DxPairs@2.0` | 結構化診斷項目 |
| E | `E.ROIs@2.0` | WSI 候選 ROI 與座標 |
| F | `F.Chunks@2.0` | 文獻檢索結果 |
| G | `G.VisualAttributeQueries@2.0` | 診斷對應的視覺屬性條件 |
| H | `H.MatchedROIs@2.0` | 視覺屬性分數與 matching 結果 |
| I | `I.CLEESelectedROIs@2.0` | CLEE 最終選擇結果 |

框架另定義 `CaseListInput@1.0` 作為 multi-case runner 的入口。若要修改元件間欄位或語意，應先更新
schema 與 canonical example，再同步上下游；不要只在 producer 中增加未宣告欄位。

## Production 與 Example Profiles

### Production

production 是無參數預設，會檢查各 config 是否真的是正式模式：

| Stage | Production config | 模式 |
|---|---|---|
| Report Decompose | `person_a/configs/report_decompose.default.json` | hospital-routed extraction |
| Interest Pattern | `person_c/configs/region_proposal.json` | CONCH + vMF region proposal |
| Knowledge Retrieval | `person_b/configs/dense.default.json` | dense retrieval |
| Query Generation | `person_a/configs/query_generation.production.json` | learnable soft prompt、single GPU |
| Visual Filter | `person_d/configs/native.json` | PLIP + CONCH、matching enabled |
| CLEE | `person_e/configs/native.json` | native CLEE |

production 強制使用 Docker executor。缺少 config、外部資產、CUDA、WSI、模型 digest 不符或輸出不合
schema 時，程序會以 non-zero exit code 停止，不會把失敗案例寫進完成 manifest。

### Example

Example profile 用合成資料檢查 contract 與編排，不載入正式模型：

```bash
mkdir -p ../run/input/pipeline
cp integration/fixtures/input/cases.example.json ../run/input/pipeline/cases.json
python pipeline/run_prepare_literature.py --profile example
python pipeline/run_pipeline.py --profile example
```

這組命令會改寫 sibling `run/input/pipeline/` 的 example artifact；若該目錄已有正式輸入，請先備份或
使用另一份 `run/` 工作目錄。Example 成功只代表 contract 與 runner 可運作，不代表 GPU 模型或正式
資產已驗證。

## Docker Images

Production runner 期待以下 image tags：

```text
wlw/person-a:0.10.0-lab23-production
wlw/person-b:0.5.0-lab23-production
wlw/person-c:0.1.0-lab23-production
wlw/person-d:0.2.0-lab23-production
wlw/person-e:1.1.0-lab23-production
```

可由專案根目錄分別建立：

```bash
docker build -f components/person_a/Dockerfile -t wlw/person-a:0.10.0-lab23-production .
docker build -f components/person_b/Dockerfile -t wlw/person-b:0.5.0-lab23-production .
docker build -f components/person_c/Dockerfile -t wlw/person-c:0.1.0-lab23-production .
docker build -f components/person_d/Dockerfile -t wlw/person-d:0.2.0-lab23-production .
docker build -f components/person_e/Dockerfile -t wlw/person-e:1.1.0-lab23-production .
```

正式流程需要 NVIDIA Container Toolkit 與可用 CUDA GPU。已驗證環境為 RTX 4090 24 GB、driver
580.178.04、94 GiB RAM；Person C 的實測最低規格以 24 GB VRAM 與 64 GB host RAM budget 為準。
不同 CUDA base image、driver 或較小 GPU 是否可用，應以各元件 README 為準，不能由 example profile
推論。

## 單獨執行元件

所有元件使用一致的 CLI 形式：

```text
python -m <module> --input <input.json> [--input <input2.json>] \
  --output <output.json> --config <config.json>
```

例如單獨執行 Person A Query Generation：

```bash
python -m components.person_a.query_generation \
  --input ../run/output/work/D_dx_pairs.json \
  --input ../run/output/work/F_chunks.json \
  --output ../run/output/work/G_queries.json \
  --config components/person_a/configs/query_generation.production.json
```

多輸入元件會依 artifact contract 辨識輸入，不依命令列順序猜測。直接在主機執行 production 元件時，
主機必須自行具備該元件的全部 Python／CUDA／system dependencies；一般建議使用對應 Docker image。

## 專案結構

```text
WLW_GPintegrate/
├── components/
│   ├── person_a/                    # Report Decompose、Query Generation
│   ├── person_b/                    # Literature Preparation、Retrieval
│   ├── person_c/                    # WSI Interest Pattern、ROI
│   ├── person_d/                    # Visual Attributes、Matching
│   └── person_e/                    # CLEE
├── contracts/
│   ├── schemas/                     # Canonical JSON Schemas
│   ├── runtime.py                   # CLI、讀寫與驗證共用程式
│   └── paths.py                     # repo/reference/run 邊界
├── integration/
│   ├── compose.yaml                 # 單案例容器拓撲
│   └── fixtures/                    # 合成整合資料
├── pipeline/
│   ├── run_prepare_literature.py
│   └── run_pipeline.py
├── tests/                           # Contract 與整合測試
└── README.md                         # 本文件
```

元件內常見的檔案包括：

- `component.yaml`：版本、入口與硬體需求摘要。
- `configs/`：example、production 或其他受控設定。
- `external-assets.yaml`：外部資產、SHA-256、revision 與授權。
- `examples/`：可散布的 canonical fixtures。
- `tests/`：元件測試。
- `PREPARATION.md`：較偏開發／交付面的檢查事項。

## 驗證與測試

執行共用測試：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

驗證 Compose 設定：

```bash
docker compose -f integration/compose.yaml config --quiet
```

各元件另有自己的測試與資產驗證命令。例如 Person C 的完整測試應在其 image 中執行，Person D 則可用
`components.person_d.verify_assets` 核對 prompts、label map、PLIP、CONCH 與 few-shot tree。

一次 production run 完成後，建議至少確認：

1. `run_manifest.json` 的 `profile` 是 `production`、`executor` 是 `docker`。
2. `completed_cases` 數量與輸入案例相同，每案有六筆 execution records。
3. D–I 的 `producer`、`case_id` 與 `reference_versions` 符合預期。
4. G 的 backend 是 `learnable_soft_prompt` 且 `device_map_strategy=single_device`。
5. H 的 `models_loaded=true` 且 `visual_matching_filter.enabled=true`。
6. I 的 `execution_statuses`、evaluated／selected／skipped 計數與 ROI `selection_history` 一致。

## 結果判讀與限制

- Pipeline exit code 0 表示所有元件與 artifact 都成功完成，不表示每個案例一定有 ROI 通過 D 或 E。
- Person D 可能在模型已實際執行後，因 `insufficient_visual_evidence`、Must condition 或 threshold 而選出
  0 個 ROI；這和 matching-disabled passthrough 是不同狀態。
- Person E 若收到 0 個 eligible ROI，會合法地回報 `skipped_no_eligible_rois`，不會為了產生非空結果而
  繞過上游門檻。
- Person C 是 CONCH + vMF 的研究型 ROI proposal，ROI 數量與外框大小會隨 tissue coverage 與分群改變。
- 全部輸出都屬研究軟體整合結果，不構成臨床診斷或模型效能認證。

正式案例的實際狀態與各 stage 時間，請以該次執行產生的 `run_manifest.json` 與對應 timestamp log
為準；不要用較早的終端輸出推定最新結果。

## 延伸文件

- [Person A README](components/person_a/README.md)
- [Person B README](components/person_b/README.md)
- [Person C README](components/person_c/README.md)
- [Person D README](components/person_d/README.md)
- [Person E README](components/person_e/README.md)
- [Person A 準備指南](components/person_a/PREPARATION.md)
- [Person B 準備指南](components/person_b/PREPARATION.md)
- [Person C 準備指南](components/person_c/PREPARATION.md)
- [Person D 準備指南](components/person_d/PREPARATION.md)
- [Person E 準備指南](components/person_e/PREPARATION.md)
