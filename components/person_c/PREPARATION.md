# 丙（Person C）：Interest Pattern／ROI Extraction 準備與交付

丙負責將單案 CaseList 的 WSI metadata 轉成 `[E]` ROI artifact。`mode: region_proposal`（`configs/region_proposal.json`）
以 SEGNET student 在 WSI 上做 c9 組織分割，取 TUM 與 BEN 的聯集（填洞）後，以每個連通物件的最小包圍框產生 ROI；
`mode: example`（`configs/example.json`）不開 WSI，只供契約與 pipeline 測試。演算法、座標規則、實測與限制見 `README.md`。
先前的 CONCH region proposal（PR #1）與 MRCPS 已移除，仍在 git 歷史。

## 1. Input／output 規格

| Entrypoint | Input | Output |
|---|---|---|
| `components.person_c.interest_pattern` | `CaseListInput@1.0`，每次一個 case | `[E] E.ROIs@2.0` |

- 必須保留 case、block、stain、filename 與 `stains[].filepath` 身分。
- ROI 座標、MPP、尺寸、路徑與 `selection_history` 必須符合 E 與共用 metadata schema。
- JSON 不嵌入 pixels；若輸出 materialized crop，路徑與檔案放在 `run/output/materialized/person_c/`。
- WSI 的外部絕對 `filepath` 是資料來源例外；checkpoint、mask、crop、cache 不得使用任意外部路徑。
- 必須在 README 定義 level-0／working-level 座標、原點、矩形邊界與縮放／rounding 規則，供丁重建 ROI。

Canonical schemas：

- `contracts/schemas/case_list_input.schema.json`
- `contracts/schemas/E_rois.schema.json`
- `contracts/schemas/metadata_case_payload.schema.json`

## 2. 交換 JSON 結構示例

以下 `<...>` 為文件省略標記。Canonical fixture 必須補齊 CaseList／metadata schema 的 required fields，
並使用可實際重建的 ROI 座標。

### Input：`CaseListInput@1.0`

```json
{
  "DxItem_list": ["Histologic_Type"],
  "case_list": [
    {
      "sample_idx": 0,
      "case_id": "case-001",
      "hospital": "EXAMPLE",
      "patient_info": {},
      "date": "",
      "source": {"fixture": "synthetic"},
      "organ": "Breast",
      "report_raw_content": "Synthetic report text.",
      "tissue_blocks": [
        {
          "block_id": "A",
          "stains": [
            {
              "stain_id": "case-001-he",
              "stain_type": "HE",
              "filename": "case-001-he.svs",
              "filepath": "/data/case-001-he.svs",
              "memo": ""
            }
          ],
          "memo": ""
        }
      ],
      "memo": ""
    }
  ]
}
```

### Output：`[E] E.ROIs@2.0`

```json
{
  "contract": "E.ROIs",
  "schema_version": "2.0",
  "artifact_id": "E-case-001",
  "case_id": "case-001",
  "producer": "person-c/interest-pattern:<version>",
  "payload": {
    "data_mode": "inference",
    "DxItem_list": ["Histologic_Type"],
    "reference_versions": {},
    "case_list": [
      {
        "sample_idx": 0,
        "case_id": "case-001",
        "hospital": "EXAMPLE",
        "patient_info": {},
        "date": "",
        "source": {"fixture": "synthetic"},
        "organ": "Breast",
        "report_raw_content": "Synthetic report text.",
        "tissue_blocks": [
          {
            "block_id": "A",
            "stains": [
              {
                "stain_id": "case-001-he",
                "stain_type": "HE",
                "filename": "case-001-he.svs",
                "filepath": "/data/case-001-he.svs",
                "memo": "",
                "roi_num": 1,
                "roi_list": [
                  {
                    "roi_id": "roi-001",
                    "global_idx": 0,
                    "local_idx": 0,
                    "level0_info": {
                      "area": 65536,
                      "coords_seg": null,
                      "cxcywh": [256, 256, 256, 256],
                      "mpp": 0.25,
                      "roi_path": null,
                      "roi_wh": [256, 256],
                      "xywh": [128, 128, 256, 256]
                    },
                    "main_info": {
                      "area": 16384,
                      "coords_seg": null,
                      "cxcywh": [128, 128, 128, 128],
                      "mpp": 0.5,
                      "roi_path": null,
                      "roi_wh": [128, 128],
                      "xywh": [64, 64, 128, 128]
                    },
                    "DxPair": null,
                    "visualAttrs": null,
                    "visualAttrs_info": null,
                    "selection_history": [
                      {
                        "stage": "interest_pattern_extraction",
                        "owner": "person_C",
                        "artifact_contract": "E.ROIs",
                        "action": "candidate_generated",
                        "status": "selected",
                        "selected": true,
                        "reason": "interest_pattern_candidate_generated",
                        "producer": "person-c/interest-pattern:<version>"
                      }
                    ]
                  }
                ]
              }
            ],
            "memo": ""
          }
        ],
        "memo": ""
      }
    ]
  }
}
```

上下游以 `case_id`、`block_id`、`stain_id` 與 `roi_id` 維持身分；丁使用 `filepath + level0_info`
重建 pixels，不應依 `global_idx` 猜測影像來源。

## 3. 打包準備

```text
WLW_GPintegrate/components/person_c/       # 程式、環境與非敏感設定
reference/person_c/checkpoint/             # ROI／segmentation 模型權重
reference/person_c/template_ref/           # 可選的 stain／label／前後處理參考
run/input/                                 # CaseList
run/output/materialized/person_c/          # 選配 ROI crop／mask
run/output/pipeline/cases/<case_id>/        # E artifact
run/cache/person_c/                         # 可刪除 cache
```

- 模型 architecture 與 WSI reader code 放 component；權重與大型 reference 放 sibling `reference`。
- patch size、stride、MPP、batch size、threshold、device 等超參數只放 config。
- 任何 external asset 均附 revision、SHA-256、license、預期 mount path，不得在程式內硬編碼工作站路徑。
- 暫存檔與 crop 必須可清理、不可覆寫 input WSI。

## 4. Unified CLI

容器介面：

```bash
component \
  --input /input/case.json \
  --output /output/E_rois.json \
  --config /config/interest_pattern.yaml
```

目前 Python／JSON config 形式：

```bash
python -m components.person_c.interest_pattern \
  --input ../run/input/pipeline/case-001.json \
  --output ../run/output/work/E_rois.json \
  --config components/person_c/configs/example.json
```

CaseList 非單案、WSI 不可讀、MPP 缺失、checkpoint/hash 錯誤、CUDA 不可用或模型失敗時，必須 non-zero
exit 且不得留下半套 E 或未完成 crop。

## 5. Dockerfile／requirements.txt

image 使用 Python 3.10、PyTorch 2.6.0 + torchvision 0.21.0（cu118）、OpenSlide；固定版本見 `requirements.txt` 與
`Dockerfile`。重建命令：

```bash
docker build --no-cache \
  -f components/person_c/Dockerfile \
  -t wlw/person-c:<version> .
```

實測（H200、單張 WSI、一個網路）：GPU 記憶體峰值 1.05 GiB、主機記憶體 7.3 GB（不畫圖）到 15.7 GB（三張圖，
最大的一張）、單張 34 到 106 秒。需要 CUDA GPU（沒有就報錯，不提供 CPU 模式）。Docker build 不下載 checkpoint，
也不把 WSI 複製進 image。**image 尚未 build**（開發環境沒有 Docker）。

## 6. Canonical example

```text
components/person_c/examples/
├── case_list.valid.json        單案 CaseList（含一張 H&E 與一張 IHC）
└── E_rois.expected.json        region_proposal 對該輸入的預期 E（由 tests 以假的切片與遮罩比對）
```

expected E 是 `region_proposal` 在假的切片與上皮遮罩（tests 內的 `canonical_output()`）上的輸出，不是真實切片的結果
（WSI 與 checkpoint 不得提交）。`region_proposal` 的行為由 `tests/` 以假的模型與分割結果覆蓋（座標、空 ROI、非處理 stain、缺檔、
缺 checkpoint non-zero fail、框的規則），並以真實 BRACS 切片實測（見 README）。

## 7. README 必填資訊

列出 component version、E schema version、模型名稱／architecture／revision、Python、WSI library、
framework、CUDA、GPU 數量、最低 VRAM、CPU fallback、checkpoint logical path 與 SHA-256、支援格式、
MPP／座標規則、CLI、Docker build/run、canonical example、效能與限制。沒有模型時需明示「無模型」。

## 8. 提交檢查

- [x] E 與每個 ROI 通過 canonical schema 及 selection semantics（tests）。
- [x] `pipeline/run_pipeline.py` 的 production 與 example 設定檢查接受 `configs/region_proposal.json` 與 `configs/example.json`。
- [x] source、configs、examples、tests、Dockerfile、requirements、README、PREPARATION 已具備。
- [x] checkpoint 只放 `reference/person_c/` 並附 manifest（SHA-256）；WSI 與 run artifacts 不進 image。
- [x] 真實 BRACS 切片以 CLI 實測（`region_proposal.json`，見 README 第 8 節）。
- [ ] production image 尚未 build 與在 pipeline 內執行（沒有 Docker）。
- [ ] 完整 pipeline（A 到 E）尚未在含 SEGNET 的版本跑通：本機缺 Person A、E 的 reference 檔，根目錄測試因此有 14 項錯誤，
      與丙無關。
- [ ] 座標／MPP round-trip 尚未和丁的 native WSI 讀取方式重測（先前的三案例整合測試是 CONCH 版本）。
- [ ] ROI 品質尚無與標註比較的數字（見 README 限制）。
