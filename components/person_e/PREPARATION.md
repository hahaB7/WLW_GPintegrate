# 戊（Person E）：CLEE Evidence Selection 準備與交付

戊負責接收 `[D]` 與 `[H]`，產生最終 `[I]`。元件必須保留 H 的全部 ROI，為每個診斷 pair 追加
可稽核的 CLEE decision；沒有 eligible ROI 是合法成功結果，不應強制載入模型。

## 1. Input／output 規格

| Entrypoint | Input | Output |
|---|---|---|
| `components.person_e.clee` | `[D] D.DxPairs@2.0` + `[H] H.MatchedROIs@2.0` | `[I] I.CLEESelectedROIs@2.0` |

- D/H/I 的 `case_id` 與 `structured_report.DxItems` 必須一致。
- 只有 H 中對相同 `dx_pair_id` 標為 selected，且 stain 屬於該診斷 pair `referenceWSI` 的 ROI 可送入模型。
- I 保留全部 H ROI 與既有 selection history，再追加 `clee` event。
- 執行模型的 ROI 必須含完整 `pseudo_DxPair`；未執行模型者不得偽造 prediction。
- `selected`、`rejected`、`skipped` 與 reason/action 必須符合中央 selection semantics。
- DRGVLM 是 I 的下游 consumer，不屬於本元件，也不得回寫或覆蓋 I。

Canonical schemas：

- `contracts/schemas/D_dx_pairs.schema.json`
- `contracts/schemas/H_matched_rois.schema.json`
- `contracts/schemas/I_clee_selected_rois.schema.json`
- `contracts/schemas/metadata_case_payload.schema.json`

## 2. 交換 JSON 結構示例

以下 `<...>` 是文件省略標記，正式 D/H/I fixture 必須補齊 canonical metadata fields。戊以
`case_id + dx_pair_id + stain_id + roi_id` 串接，不得依陣列順序配對。

### First input：`[D] D.DxPairs@2.0`

```json
{
  "contract": "D.DxPairs",
  "schema_version": "2.0",
  "artifact_id": "D-case-001",
  "case_id": "case-001",
  "producer": "person-a/report-decompose:<version>",
  "payload": {
    "data_mode": "inference",
    "DxItem_list": ["Histologic_Type"],
    "case_list": [
      {
        "case_id": "case-001",
        "structured_report": {
          "DxItems": {
            "Histologic_Type": {
              "dx_pair_id": "case-001-dx-001",
              "DxResultCls": "<case result>",
              "referenceWSI": ["case-001-he"],
              "<其餘 DxItem fields>": "見 metadata schema"
            }
          }
        },
        "<其餘 metadata-shaped case fields>": "見 metadata schema"
      }
    ]
  }
}
```

### Second input：`[H] H.MatchedROIs@2.0`

```json
{
  "contract": "H.MatchedROIs",
  "schema_version": "2.0",
  "artifact_id": "H-case-001",
  "case_id": "case-001",
  "producer": "person-d/visual-filter:<version>",
  "payload": {
    "data_mode": "inference",
    "DxItem_list": ["Histologic_Type"],
    "case_list": [
      {
        "case_id": "case-001",
        "structured_report": {"<same DxItems as D>": "..."},
        "tissue_blocks": [
          {
            "block_id": "A",
            "stains": [
              {
                "stain_id": "case-001-he",
                "roi_num": 1,
                "roi_list": [
                  {
                    "roi_id": "roi-001",
                    "level0_info": {"<ROI geometry>": "..."},
                    "main_info": {"roi_path": null, "<其餘 ROI geometry>": "..."},
                    "selection_history": [
                      "<person C event>",
                      {
                        "stage": "visual_attributes_matching_filter",
                        "owner": "person_D",
                        "artifact_contract": "H.MatchedROIs",
                        "action": "legality_evaluated",
                        "status": "selected",
                        "selected": true,
                        "reason": "visual_attributes_match",
                        "producer": "person-d/visual-filter:<version>",
                        "dx_pair_id": "case-001-dx-001",
                        "query_id": "query-001"
                      }
                    ],
                    "<其餘 ROI fields>": "見 metadata schema"
                  }
                ],
                "<其餘 stain fields>": "見 metadata schema"
              }
            ],
            "memo": ""
          }
        ],
        "<其餘 case fields>": "見 metadata schema"
      }
    ]
  }
}
```

### Output：`[I] I.CLEESelectedROIs@2.0`

```json
{
  "contract": "I.CLEESelectedROIs",
  "schema_version": "2.0",
  "artifact_id": "I-case-001",
  "case_id": "case-001",
  "producer": "person-e/clee:1.1.0",
  "payload": {
    "data_mode": "inference",
    "DxItem_list": ["Histologic_Type"],
    "reference_versions": {
      "clee_inference": {
        "backend": "native",
        "<checkpoint/threshold provenance>": "見 canonical output"
      }
    },
    "case_list": [
      {
        "case_id": "case-001",
        "structured_report": {"<same DxItems as D/H>": "..."},
        "tissue_blocks": [
          {
            "block_id": "A",
            "stains": [
              {
                "stain_id": "case-001-he",
                "roi_num": 1,
                "roi_list": [
                  {
                    "roi_id": "roi-001",
                    "pseudo_DxPair": {
                      "0": {
                        "Histologic_Type": {
                          "chunk_idx": 0,
                          "assigned": "<class>",
                          "importance": 0.82,
                          "assigned_as_ref": true,
                          "candidate@top3AUC": {"<class>": 0.8},
                          "candidate@ALL": {"<class>": 0.8}
                        }
                      },
                      "finalResult": {
                        "Histologic_Type": {
                          "chunk_idx": 0,
                          "assigned": "<class>",
                          "importance": 0.82,
                          "assigned_as_ref": true,
                          "candidate@top3AUC": {"<class>": 0.8},
                          "candidate@ALL": {"<class>": 0.8}
                        }
                      }
                    },
                    "selection_history": [
                      "<unchanged person C event>",
                      "<unchanged person D event>",
                      {
                        "stage": "clee",
                        "owner": "person_E",
                        "artifact_contract": "I.CLEESelectedROIs",
                        "action": "evidence_evaluated",
                        "status": "selected",
                        "selected": true,
                        "reason": "case_importance_threshold_met",
                        "producer": "person-e/clee:1.1.0",
                        "backend": "native",
                        "dx_pair_id": "case-001-dx-001",
                        "threshold": 0.7,
                        "comparison": ">=",
                        "score": 0.82
                      }
                    ],
                    "<其餘 ROI geometry/visual fields>": "完整沿用 H"
                  }
                ],
                "<其餘 stain fields>": "完整沿用 H"
              }
            ],
            "memo": ""
          }
        ],
        "<其餘 case fields>": "完整沿用 H"
      }
    ]
  }
}
```

若 H 沒有 eligible ROI，I 仍保留 ROI，但 CLEE event 使用 `action=inference_skipped`、
`status=skipped`，且不得加入 `pseudo_DxPair`。

## 3. 打包準備

```text
WLW_GPintegrate/components/person_e/       # CLEE adapter、native backend、環境與 configs
reference/person_e/checkpoint/             # MedGemma、embedding、CLEE prefix/checkpoint、threshold bundle
reference/person_e/template_ref/           # fixture label space 與測試 threshold
run/output/pipeline/cases/<case_id>/        # D、H、I
run/output/materialized/person_e/           # 選配 ROI materialization
run/cache/person_e/                         # external-command/native 暫存
```

- `configs/default.json` 是 deterministic fixture，只供 contract／pipeline 測試。
- `configs/native.json` 是 native inference 設定；部署前由 `native.example.json` 複製並填入可驗證路徑。
- checkpoint active label space 是 authority；deployment whitelist 只能縮小，不能擴張 checkpoint 支援範圍。
- 正式設定固定使用 `clee-261027221211` epoch 223 與同 epoch 的 validation-calibrated threshold bundle。
- `DxItem_embeddings_2607231639.pt/` 是含 `embedding_manifest.json` 的唯讀目錄，不再是單一 `.pt` 檔。
- evidence retention 使用 `case_importance.by_DxResult`；個別 DxResult 不可校準時才記錄原因並降級至 `by_DxItem`，不得挪用 `roi_classification` threshold。
- 零候選 policy 的 scope 是 case × DxItem；raw retained count 為 0 時，只在 terminal layer 依 evidence score 降冪、stable `global_idx` 破同分，取 `fallback_top_k`（預設 1）。
- 模型、embedding、checkpoint、threshold 本體不進 Git 或 image；在 `external-assets.yaml` 記錄 revision、
  SHA-256、license 與 mount path。
- D/H/I、ROI crop、backend 暫存與 log 一律位於 sibling `run/`。

## 4. Unified CLI

容器介面：

```bash
component \
  --input /input/D_dx_pairs.json \
  --input /input/H_matches.json \
  --output /output/I_selected_rois.json \
  --config /config/clee.yaml
```

專案 fixture 命令：

```bash
python -m components.person_e.clee \
  --input ../run/output/pipeline/cases/case-001/D_dx_pairs.json \
  --input ../run/output/pipeline/cases/case-001/H_matches.json \
  --output ../run/output/work/I_selected_rois.json \
  --config components/person_e/configs/default.json
```

正式 native inference 將 config 改為 `components/person_e/configs/native.json`。D/H case 不一致、資產缺失、
checkpoint/threshold epoch 不相容、CUDA 不可用、backend coverage 不完整或輸出違反 schema 時必須
non-zero exit，不得寫出部分 I。

## 5. Dockerfile／requirements.txt

目前 component manifest 的正式需求為：

- Component version：`1.1.0`
- Python：`3.10`
- CUDA runtime：`11.8`
- GPU：必要，至少 1 張
- 最低 VRAM：16 GB
- 建議 CPU：4 cores；RAM：32 GB；timeout：3600 秒
- 主要模型：`medgemma-1.5-4b-it`
- CLEE checkpoint：由 `configs/native.json:checkpoint_dir` 指定

以上是目前 manifest 的交付基線；若實測不同，必須同步修改 manifest 與 README，不可只改 config。

```bash
docker build --no-cache \
  -f components/person_e/Dockerfile \
  -t wlw/person-e:1.1.0 .
```

clean-machine 測試需包含 fixture CPU path 與 native GPU path。Docker build 不得下載私有模型或 checkpoint；
執行時以唯讀 `/reference` mount 提供。

## 6. Canonical examples

至少交付兩組：

```text
components/person_e/examples/
├── fixture/
│   ├── D_dx_pairs.valid.json
│   ├── H_matches.valid.json
│   ├── I_selected_rois.expected.json
│   ├── config.example.json
│   └── README.md
└── native-smoke/
    ├── D_dx_pairs.valid.json
    ├── H_matches.valid.json
    ├── expected_structure.json
    └── README.md
```

fixture expected output 必須 byte-stable。native smoke 可因硬體數值誤差只驗證 schema、ROI coverage、event
semantics 與 score 容許誤差，但需記錄 GPU/CUDA/framework。範例至少覆蓋 selected、rejected、
upstream skipped、空 eligible set 與多 DxItem 合併。

## 7. README 必填資訊

列出 component/version、D/H/I schema、MedGemma 與 CLEE checkpoint revision、embedding／threshold bundle、
Python/PyTorch/Transformers/CUDA、GPU 數量、最低 VRAM、CPU fixture 支援、資產 logical path 與 SHA-256、
label-space/threshold authority、chunk 行為、CLI、Docker build/run、canonical example、錯誤條件與限制。

## 8. 提交檢查

- [ ] I 通過 schema 與 selection semantics，H ROI 全數保留。
- [ ] fixture、空 eligible、multi-DxItem、native GPU smoke 與錯誤路徑皆通過。
- [ ] checkpoint epoch、threshold bundle、label space 與 component version 可相互驗證。
- [ ] `by_DxResult` resolution／`by_DxItem` fallback 與 terminal-layer Top-K audit 可由 `reference_versions.clee_inference` 重建。
- [ ] clean-machine image 可跑 fixture 與掛載 reference 後的 native smoke。
- [ ] source、configs、examples、tests、Dockerfile、requirements、README、PREPARATION 已提交。
- [ ] 模型與 checkpoint 只放 `reference/person_e/`；run artifacts、cache、crop 與 credential 不提交。
- [ ] 根目錄 contract 與完整 pipeline tests 全部通過。
