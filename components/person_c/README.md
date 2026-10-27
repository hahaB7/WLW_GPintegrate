# 丙(Person C):WSI Interest Pattern Extraction

把一個 case 的 WSI 轉成 ROI 清單 `[E]`,交給丁(`visual_filter`)。

- Owner:person_C
- Component version:`0.1.0`(`status: integration_draft`,第一版可跑通,演算法為初版)
- 輸入:`CaseListInput@1.0`(每次一個 case)
- 輸出:`E.ROIs@2.0`
- 入口:`python -m components.person_c.interest_pattern --input ... --output ... --config ...`

## 1. 在整個流程中的位置

```text
                      WLW pipeline(單一 case)

  CaseList ──> 甲 report_decompose ──> [D] ──┬──> 乙 knowledge_retrieval ──> [F]
      │                                      │                                │
      │                                      │         甲 query_generation <──┘
      │                                      │                  │
      │                                      │                 [G]
      │                                      │                  │
      └──> 丙 interest_pattern ──> [E] ──────┼──> 丁 visual_filter <── [G]
           ^^^^^^^^^^^^^^^^^^^^^               │            │
           本元件(這份 README)                │           [H]
                                               │            │
                                               └──> 戊 clee <┘ ──> [I] ──> DRGVLM
```

丙只依賴 CaseList,不依賴甲、乙;丁、戊之後會讀丙的 ROI 座標去重建影像。

## 2. 這個元件做什麼

```text
 CaseList JSON (一個 case)
        │
        ▼
 ┌────────────────────────────────────────────────────────────┐
 │ interest_pattern.py                                        │
 │                                                            │
 │  驗證輸入(單案、schema)                                  │
 │        │                                                   │
 │        ▼                                                   │
 │  for 每個 block ─> for 每個 stain                          │
 │        │                                                   │
 │        ├─ stain_type 不在 stain_types ──> roi_list = []    │
 │        │                                                   │
 │        └─ 要處理的 stain:                                  │
 │             │                                              │
 │             ▼                                              │
 │     ┌───────────────┐   讀 slide 尺寸與 MPP                │
 │     │ slide_info    │   (沒有 MPP 就直接失敗)             │
 │     └──────┬────────┘                                      │
 │            ▼                                               │
 │     ┌───────────────────────────────────────────┐          │
 │     │ region/proposal.py   (GPU)                │          │
 │     │                                           │          │
 │     │  CONCH 抽 dense 特徵                      │          │
 │     │     ──> vMF 分群 (K 群)                   │          │
 │     │     ──> zero-shot 幫每群命名 (BCSS 詞表)  │          │
 │     │     ──> 只留 proposal 類別                │          │
 │     │            tumor / dcis /                 │          │
 │     │            normal_acinus_or_duct /        │          │
 │     │            metaplasia_NOS                 │          │
 │     │     ──> 輸出 polygon (level-0 像素)       │          │
 │     └──────────────────┬────────────────────────┘          │
 │                        ▼                                   │
 │     ┌───────────────────────────────────────────┐          │
 │     │ rois.py   (純 Python)                     │          │
 │     │  每個聯通元件 ──> 一個外接矩形 ROI        │          │
 │     │  level0_info + main_info (換成 target_mpp)│          │
 │     └──────────────────┬────────────────────────┘          │
 │                        ▼                                   │
 │          roi_list / roi_num / selection_history            │
 │                                                            │
 │  寫檔前再驗證 E.ROIs 契約,失敗就不寫檔                    │
 └────────────────────────────────────────────────────────────┘
        │
        ▼
 E.ROIs JSON  ──>  丁 visual_filter
```

重點:

- **不輸出圖片。** `roi_path` 為 `null`,下游用 `stains[].filepath` 加 `level0_info` 自己從 WSI 裁圖。
- **`region/` 是可替換的部分。** 它只負責「給一張 WSI,回傳 polygon」;之後換演算法,只要保持這個介面,CLI、契約驗證、ROI 轉換都不用動。

## 3. 兩種模式

| Config | 行為 |
|---|---|
| `configs/example.json`(`mode: example`) | 契約用的空殼,不開 slide,ROI 為空。共用 pipeline 和既有測試使用 |
| `configs/region_proposal.json`(`mode: region_proposal`) | 上圖的真實流程 |

## 4. 座標與 ROI 規則(下游以本節為準)

- 座標單位是 level-0 像素,原點在左上,x 向右、y 向下。
- `xywh = [left, top, width, height]`,涵蓋 `[left, left+width) × [top, top+height)`,一定在 slide 範圍內。
- **一個聯通元件(一個 proposal polygon)一個 ROI**,取其外接矩形(往外取整到整數像素,夾在 slide 範圍內)。ROI 大小不固定;巢狀或相鄰元件的外框可能重疊。
- 小於 `roi.min_region_area_px` 的元件丟掉(面積已扣除挖洞);每個 stain 最多 `roi.max_rois_per_stain` 個,由大的元件先取。
- 排序為 `(top, left)`,`global_idx` 依「block → stain → 此順序」從 0 連續編號;`local_idx` 是 stain 內序號。
- `roi_id = <stain_id>-roi-<local_idx:05d>`,case 內唯一。
- `main_info` 是同一個矩形換算到「每個 ROI 自己的工作 MPP」:`mpp = max(roi.target_mpp, 最長邊(µm) ÷ roi.max_main_side_px)`。也就是預設用 `target_mpp`,若這樣會讓最長邊超過 `max_main_side_px`(預設 2048),就改用更粗的 MPP,大的聯通元件因此是縮小看,不會被切開。`level0_info` 完全不受影響。
  `roi_wh = max(1, round(level0_wh × level0_mpp ÷ mpp))`(每軸),`xywh` 以同一比例縮放並四捨五入。這與 person_e 讀 ROI 的公式一致。
- `mpp` 欄位:x、y 相同時是單一數字,否則是 `[mpp_x, mpp_y]`。
- `coords_seg`、`DxPair`、`visualAttrs`、`visualAttrs_info` 都是 `null`。
- 每個 ROI 附一筆 `selection_history` 事件:`stage=interest_pattern_extraction`、`action=candidate_generated`、`status=selected`。

### 4.1 丙對下游的保證

對任何成功輸出的 `E.ROIs`,以下成立(第 1、2、4、5、6、7 項由單元測試與 `examples/` 檢查;第 3 項由程式內的 MPP 檢查保證;第 8 項是幾何特性,無法測試):

1. 同一個 case 內,`roi_id` 與 `global_idx` 各自唯一;`roi_num == len(roi_list)`。
2. 每個 ROI 的 `level0_info.xywh` 為整數、寬高 > 0,且完全落在 slide 的 level-0 範圍內。
3. `level0_info.mpp` 是 slide 的 level-0 MPP(來自 `openslide.mpp-x/y`,缺失即失敗),`main_info.mpp` 是該 ROI 的工作 MPP(單一數字,≥ `roi.target_mpp`)。
4. `main_info.roi_wh = max(1, round(level0_info.roi_wh × level0_info.mpp ÷ main_info.mpp))`(每軸);`main_info.xywh` 是 `level0_info.xywh` 以同一比例換算。
5. `roi_path` 一律為 `null`(不輸出圖片);`coords_seg`、`DxPair`、`visualAttrs`、`visualAttrs_info` 一律為 `null`。
6. `selection_history` 第一筆是丙的事件(`stage=interest_pattern_extraction`、`status=selected`);丙不會寫 `pseudo_DxPair`。
7. 非處理的 stain 照常保留,`roi_num=0`、`roi_list=[]`;一個 case 沒有任何 ROI 也是合法的成功輸出。
8. 外接矩形可能互相重疊、可能包含空白;同一個 ROI 不保證整塊都是病灶。

### 4.2 下游如何取得 ROI 影像

不要用 `global_idx` 猜來源。以 `stain.filepath`(外部絕對路徑)加 `level0_info` 重建:

```python
import openslide
slide = openslide.OpenSlide(stain["filepath"])
x, y, w, h = roi["level0_info"]["xywh"]          # level-0 像素,左上角起算
out_w, out_h = roi["main_info"]["roi_wh"]         # 影像最終尺寸
# 從夠細的金字塔層讀 (x, y, w, h) 這個 level-0 區域,縮放到 (out_w, out_h) 即為 main_info 視角的 ROI。
```

這與 person_e 的 `_read_wsi_roi` 使用同一個公式,因此戊可以直接用,不需要額外轉換。

已驗證:用 person_e 的 `_read_wsi_roi` 實際讀取 BRACS_1003691 的 16 個 ROI(含最大的 6 個,最長邊 > 12000 px),讀出的影像尺寸與 `main_info.roi_wh` 全部一致。

## 5. 設定(`configs/region_proposal.json`)

| 鍵 | 說明 |
|---|---|
| `stain_types` | 要處理的染色類型,`null` 代表全部。其他 stain 保留,`roi_list=[]` |
| `region.conch_checkpoint`、`vocab_json` | 外部資產(權重、BCSS 類別描述),必須在 `reference/person_c/` 之下;執行前檢查 SHA-256 |
| `region.conch_lib_path` | 隨元件發佈(`third_party/CONCH`,相對於 `components/person_c/`);同樣檢查 SHA-256 |
| `region.vocab`、`proposal_categories` | zero-shot 詞表(`BCSS`)與保留的類別,`[]` 代表詞表預設 |
| `region.fields`、`mag`、`overlap` | CONCH 視野(level-0 px)、倍率、重疊比例 |
| `region.K`、`radius`、`pca_dim`、`use_patch`、`seed`、`zs_temp` | 分群與 zero-shot 參數 |
| `region.device`、`batch_size`、`num_workers` | 執行參數 |
| `roi.target_mpp`、`max_main_side_px`、`min_region_area_px`、`max_rois_per_stain` | ROI 轉換參數(`max_main_side_px`:`main_info` 最長邊上限) |

## 6. 執行

目錄必須是 repo 的同層結構(見根目錄 README):`../reference/person_c/`、`../run/`。

```bash
python -m components.person_c.interest_pattern \
  --input  ../run/input/pipeline/case-001.json \
  --output ../run/output/work/E_rois.json \
  --config components/person_c/configs/region_proposal.json
```

失敗(非單案、slide 讀不到、MPP 缺失、CUDA 不可用、模型錯誤、輸出不合契約)一律非零結束,不寫出 E。

容器(在 repo 根目錄 build;`reference/`、`run/` 以掛載提供,不放進 image):

```bash
docker build -f components/person_c/Dockerfile -t wlw/person-c:0.1.0 .
docker run --rm --gpus all \
  -v "$(pwd)/../reference:/reference:ro" -v "$(pwd)/../run:/run" \
  wlw/person-c:0.1.0 \
    --input  /run/input/pipeline/case-001.json \
    --output /run/output/work/E_rois.json \
    --config /app/components/person_c/configs/region_proposal.json
```

(image tag 與 `component.yaml` 的 `version`、artifact 的 `producer` 一致:`person-c/interest-pattern:0.1.0`。)

測試(純 Python,不需要 GPU):

```bash
python -m unittest components.person_c.tests.test_interest_pattern
```

含 `examples/` 的 canonical example 比對、輸入錯誤(多案、slide 不存在)與資產 digest 檢查(通過、checkpoint 被改、library 被改、檔案缺失)測試。測試只用 `components/person_c/examples/` 內的檔案(含 `cases.example.json`),image 內不需要 `integration/` 即可在乾淨環境完整執行:

```bash
docker run --rm --entrypoint python wlw/person-c:0.1.0 -m unittest components.person_c.tests.test_interest_pattern
```

端到端(真實 WSI → CONCH → ROI,需 GPU 與資產;步驟與實測結果見 `examples/README.md`):

```bash
SLIDE=/path/to/slide.svs PYTHON=python bash components/person_c/examples/e2e/run_e2e.sh
```

## 7. 環境與外部資產

| 項目 | 內容 |
|---|---|
| 模型 | CONCH `conch_ViT-B-16`(權重 SHA-256 與授權見 `external-assets.yaml`) |
| Python | 3.12 |
| 主要套件 | torch 2.14.0+cu130(CUDA 13.0)、torchvision 0.29.0、timm 1.0.29、openslide-python 1.4.6、scipy 1.18.1、numpy 2.5.3(完整版本見 `requirements.txt`) |
| CONCH 程式庫 | 隨元件發佈於 `third_party/CONCH/`:上游 `mahmoodlab/CONCH` commit `141cc09c7d4ff33d8eda562bd75169b457f71a62` 的 `conch/open_clip_custom/` 與 `LICENSE`,逐檔未修改(CC-BY-NC-ND 4.0);目錄 tree SHA-256 `9e6419e1c78f79406acf635529bfde1c4029f15acb97221d11e76345d187febd` |
| 資產 SHA-256 | 權重 `pytorch_model.bin`:`40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c`;`bcss_vocab.json`:`47ea2b920bb73dfd43e9f3b9892fab20b80e6812902a9b914b6986f13166aa62`。數值固定在 `assets.py`,執行時連同 `third_party/CONCH` 一併比對,不符即 `AssetError` 並非零結束,不寫出 E |
| 外部資產位置 | 權重:`reference/person_c/checkpoint/conch/pytorch_model.bin`;類別描述:`reference/person_c/template_ref/bcss_vocab.json` |
| 資產準備 | 1) 在 https://huggingface.co/MahmoodLab/CONCH 接受條款後下載 `pytorch_model.bin`(CC-BY-NC-ND,gated,不可轉散布)。2) `bcss_vocab.json`(專案自有、無使用限制)由 Drive 的 `WLW_GPintegrate Reference Assets` 提供,放在上述位置。兩者 SHA-256 與上表不符會直接失敗。CONCH 程式庫已在 repository 內,不需準備 |
| GPU | 必要,至少 1 張;沒有 CPU fallback(`device: cuda` 但 CUDA 不可用會直接失敗)。**主機 NVIDIA driver 須 ≥ 580**(CUDA 13.0 的最低要求;低於此版本容器會在啟動階段被 `cuda>=13.0` 條件拒絕,例如 driver 535 的 RTX 4090 主機)。Docker 基底為 `nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04`(沿用 person_e 的做法)。若目標環境 driver < 580,需改用 CUDA 12.x 基底(如 `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`,driver ≥ 570)並改用對應 `cu128` 的 torch wheel;**此組合未 build、未驗證**。 |

## 8. 實測(初版,單張 slide)

| 項目 | 數值 |
|---|---|
| 測試 slide | BCSS 公開資料 `TCGA-AC-A6IW`,28777 × 34620 px,MPP 0.252 |
| 環境 | Nano4 `dev` 佇列,1 張 H200,8 CPU |
| 總時間 | 1 分 13 秒 |
| GPU 記憶體 | 約 4.7 GB(PyTorch allocated,特徵抽取階段) |
| CPU 記憶體(peak RSS) | 約 10.9 GB |
| 輸出 | **2 個 ROI**,通過 schema 與 selection 語意檢查,全部在 slide 範圍內 |
| ROI 內容 | 一個覆蓋大片腫瘤的外框(level-0 25664 × 29248 px,約佔 slide 75%;`main_info` 12935 × 14741 px),一個小外框(704 × 960 px) |

再量一張大的:BRACS test `BRACS_1003691`(Group_AT / ADH),61879 × 70581 px(約 4.4 億像素,檔案 1.3GB),MPP 0.2524,同環境。

| 項目 | 數值 |
|---|---|
| 總時間 | 3 分 36 秒 |
| CPU 記憶體(peak RSS) | 約 42.6 GB |
| 輸出 | 160 個 ROI(未達 256 上限),通過 schema 與 selection 語意檢查,全部在 slide 範圍內 |
| ROI 大小(level-0 最長邊) | 中位數 1632 px,最小 416 px,最大 14080 px |
| 佔 slide 面積 | 中位數 0.04%,最大 4.5% |
| `main_info` 最長邊 > 4096 px 的 ROI | 4 個(最大 7108 px) |

> 註:以上 ROI 大小是加入 `max_main_side_px` 之前量到的。加入後,`main_info` 最長邊上限為 2048 px,以重算驗證:BCSS smoke 從 14741 降到 2048,BRACS_1003691 從 7108 降到 2048;`level0_info` 不變。

目視疊圖:框落在有組織處,背景沒有框。但曲折的組織帶外框內會包含很多空白(外接矩形的特性)。兩張 slide 的時間與記憶體隨 slide 大小增加;更大的 slide 尚未量測。

## 9. 已知限制與待辦

- **演算法是初版。** `region/` 目前是 CONCH 特徵分群加 zero-shot 命名,尚未針對 WLW 的診斷項目調整,之後會替換。
- **外接矩形可能非常大。** 腫瘤連成一個聯通元件時,`level0_info` 的外框會涵蓋大部分 slide(見第 8 節)。已用 `roi.max_main_side_px` 限制 `main_info` 的最長邊(大的 ROI 縮小看),但 `level0_info` 仍是完整外框,且外框內可能包含很多空白。是否要切分大元件,需與負責人、戊討論。
- **ROI 數量依賴分群結果。** 外框是每個元件一個,元件很少時 ROI 很少,元件很碎時很多;`max_rois_per_stain`(預設 256)只在後者生效。
- **ROI 規則以第 4 節為準。** 下游依第 4.1、4.2 節使用;規則變更時需更新 README 與 `component.yaml` 版本。
- **只處理 `stain_types` 內的 stain**,預設只有 `HE`。
- **zero-shot 詞表是 BCSS 乳腺組織。** 其他器官需要換詞表。
- **Dockerfile 已寫好,但容器本身尚未 build。** 這台叢集沒有 Docker,Apptainer 的 `--fakeroot` build 在這裡也不能用(帳號不在 `/etc/subuid`,root 映射命名空間讀不到 `/work`、`$HOME`)。已驗證的是 Dockerfile 的 pip 步驟:用全新的 Python 3.12 venv,依序執行 Dockerfile 的兩個安裝指令(cu130 版 torch 與 torchvision,再 `pip install -r requirements.txt`),跑 smoke case,ROI 幾何與原生環境完全一致。**未驗證**的是 `nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04` 基底(tag 已確認存在於 Docker Hub)、`apt`、`docker build` 本身、容器內掛載 `/reference` 與 `/run`。請在有 Docker 的機器上跑一次 `docker build` 與 `docker run`。
- **VRAM、RAM、timeout 只量了兩張 slide。** 最大的一張(61879 × 70581 px)需約 14.6 GB VRAM、峰值 45.7 GB RAM、5 分 4 秒;`component.yaml` 填的是這些實測值,更大的 slide 尚未量測。24 GB 顯示卡可以容納 14.6 GB 的 VRAM 需求,但仍需要約 46 GB 以上的主機記憶體。
- **尚未與丁、戊的真實實作做 round-trip 整合測試**(目前他們也是 stub 或 fixture)。
- 輸出在固定 seed 下分群可重現,但跨 GPU 型號的數值差異未驗證。

## 10. 檔案

```text
components/person_c/
├── interest_pattern.py       入口:CLI、stain 迴圈、組裝 E.ROIs
├── rois.py                   polygon -> ROI 矩形、座標與 MPP 換算
├── region/
│   ├── proposal.py           一張 WSI -> proposal polygon(GPU)
│   ├── preprocess.py         CONCH dense 特徵抽取(複製自 PIPELINE/ROI/utils)
│   ├── partition.py          vMF 分群、polygon 輸出(同上)
│   └── zeroshot.py           CONCH 文字端 zero-shot 命名(同上)
├── configs/                  example.json、region_proposal.json、README.md
├── third_party/CONCH/        上游 CONCH 的 open_clip_custom 與 LICENSE(未修改)
├── assets.py                 資產 SHA-256 固定值與執行前檢查
├── examples/                 canonical example(case、proposal polygon、預期 E)與 e2e/ 端到端腳本
├── tests/                    test_interest_pattern.py
├── component.yaml            版本與環境需求
├── external-assets.yaml      外部資產清單(SHA-256、授權)
├── requirements.txt
├── Dockerfile                (nvidia/cuda 13.0.3 + venv;pip 步驟已驗證,容器未 build)
├── PREPARATION.md            交付檢查表(上游提供)
└── README.md                 本文件
```

## 11. 變更紀錄

| 版本 | 內容 |
|---|---|
| 0.1.0(review 第一輪修正) | 測試改用 `examples/` 內的 fixture;CONCH 固定到 `141cc09`;執行前檢查 checkpoint、library、詞表的 SHA-256;詞表改隨元件發佈;新增端到端腳本與 driver 需求說明。 |
| 0.1.0 | 第一版可跑通:CONCH 區域提案(`region/`)加聯通元件外接矩形 ROI(`rois.py`),`mode: region_proposal`;`mode: example` 保留契約空殼。ROI 的 `main_info` 最長邊上限 `roi.max_main_side_px`。已在 2 張真實 slide 上驗證(第 8 節)。 |
