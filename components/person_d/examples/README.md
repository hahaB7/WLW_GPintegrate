# 丁（Person D）canonical example

`visual_filter` 的單案範例：`[E] E.ROIs@2.0` + `[G] G.VisualAttributeQueries@2.0` →
`[H] H.MatchedROIs@2.0`。

| 檔案 | 內容 |
|---|---|
| `E_rois.valid.json` | case `case-001`；`case-001-he`（參考切片，指向 `synthetic_slide.tiff`）與 `case-001-er`（非參考切片）各 2 個 ROI |
| `E_rois.empty.valid.json` | 同一 case，所有 stain 皆無 ROI |
| `G_queries.valid.json` | 完全合成的 criteria fixtures：`query-001`／`002` 驗證 Must condition；`query-004` 將可確認選項設為 High_Possibly_True；`query-005` 設為 Low_Possibly_True；`query-003` 為 unmapped。|
| `synthetic_slide.tiff` | OpenSlide generic tiled TIFF，33792 × 19968 px，mpp 0.5；只有兩個 HE ROI 範圍內有程式產生的仿 H&E 紋理，其餘為白底；SHA-256 `a1f9c031b72c67308a860e24ba3f1a6fa2a04e00564a6990322ba47e22eb3d30` |
| `H_matches.expected.json` | `example` 模式完整預期輸出（逐內容比對） |
| `H_matches.empty.expected.json` | `native` 模式、空 ROI case 的完整預期輸出；不載入模型（逐內容比對） |
| `H_matches.native.structure.json` | `native` 模式、完整 case 的預期結構 |
| `check_native_structure.py` | 依結構檔檢查 `native` 輸出 |
| `person_c_integration/` | 以丙（#1）的 E 執行丁 native 所需的 fixture 產生程式與規格 |

## 涵蓋情境

| 情境 | 位置 | canonical run 的判定 |
|---|---|---|
| 四種 informative conditions | 兩個 HE ROI 的 `query-001`、`002`、`004`、`005` | 任一模型命中 Must_True、High_Possibly_True、Low_Possibly_True 或 Must_False 即固定 `selected`／`informative_visual_attributes_forwarded` |
| 非 reference WSI（skipped） | `case-001-er` 的所有 ROI × 所有 query | 固定 `skipped`／`stain_not_in_reference_wsi`，每個 query 一筆並帶 `query_id` |
| unmapped query（skipped） | `case-001-query-003` | 固定 `skipped`／`query_unmapped` |
| 空 ROI case | `E_rois.empty.valid.json` | 逐內容比對 |

`query-004`、`query-005` 的 criteria 由 `query-001` 衍生：每個屬性的選項分別全設為 High_Possibly_True
或 Low_Possibly_True（`Unable to confirm` 設為 Not_Mentioned）。`query-001`、`query-002` 使用專為
matching 測試設計的合成 Must condition matrix。Production policy 不使用 condition 正負、Must gate 或
score 作選送門檻；只要 PLIP 或 CONCH 任一模型在至少一個屬性提供 informative evidence 即往 Person E
送。這些 condition matrix 不是任何診斷類別的 ground truth。
各分支以假模型做的確定性覆蓋另見 `components/person_d/tests/`。

## 來源與去識別化

- E 取自丙提供的合成樣本：移除 `roiCase` schema 不允許的 `structured_report`，並將 `case-001-he`
  的 `filename`／`filepath` 改指向合成切片。
- G 的 visual attribute 結構來自 condition-free inference template；所有
  `diagnosticCriteria.conditions`、case metadata、ID 與 provenance 均為合成 fixture，
  不含或引用診斷 ground truth。
- 合成切片由程式產生，不含任何影像來源或 PHI。

## 執行

CLI 只接受 sibling `run/` 下的輸入；`E_rois.valid.json` 的切片路徑
`../run/input/person_d/examples/synthetic_slide.tiff` 也依此複製位置設定：

```bash
mkdir -p ../run/input/person_d/examples
cp components/person_d/examples/E_rois.valid.json \
   components/person_d/examples/E_rois.empty.valid.json \
   components/person_d/examples/G_queries.valid.json \
   components/person_d/examples/synthetic_slide.tiff \
   ../run/input/person_d/examples/

# 1. example 模式：不需外部資產
python -m components.person_d.visual_filter \
  --input ../run/input/person_d/examples/E_rois.valid.json \
  --input ../run/input/person_d/examples/G_queries.valid.json \
  --output ../run/output/person_d/examples/H_matches.json \
  --config components/person_d/configs/example.json

# 2. native 模式、空 ROI case：需 reference/person_d/template_ref 的 prompt 與 label map
python -m components.person_d.visual_filter \
  --input ../run/input/person_d/examples/E_rois.empty.valid.json \
  --input ../run/input/person_d/examples/G_queries.valid.json \
  --output ../run/output/person_d/examples/H_matches.empty.json \
  --config components/person_d/configs/native.example.json

# 3. native 模式、完整 case：需 PLIP、CONCH 與 Example 範例圖
#    尚無 CONCH 授權時改用 configs/native.smoke.json（隨機權重，只驗證流程）
python -m components.person_d.visual_filter \
  --input ../run/input/person_d/examples/E_rois.valid.json \
  --input ../run/input/person_d/examples/G_queries.valid.json \
  --output ../run/output/person_d/examples/H_matches.native.json \
  --config components/person_d/configs/native.json
```

`contracts/paths.py` 只允許 `components/person_x/configs/` 下的 config，因此範例直接使用
`configs/` 內的設定，不另放 `config.example.json`。Docker 執行方式見 `components/person_d/README.md`。

## 比對

```bash
compare() { python -c "import json, sys; a, b = (json.load(open(p, encoding='utf-8')) for p in sys.argv[1:]); sys.exit(a != b)" "$1" "$2" && echo "match: $2"; }
compare components/person_d/examples/H_matches.expected.json ../run/output/person_d/examples/H_matches.json
compare components/person_d/examples/H_matches.empty.expected.json ../run/output/person_d/examples/H_matches.empty.json
python components/person_d/examples/check_native_structure.py \
  components/person_d/examples/H_matches.native.structure.json \
  ../run/output/person_d/examples/H_matches.native.json
```

- 1、2 以 JSON 內容逐一比對；2 的 `reference_versions` 含 prompt 與 label map 的 SHA-256，
  更新 `template_ref` 資產後需重新產生。
- 3 的模型相似度分數取決於模型與 GPU 浮點誤差，不與固定輸出逐值比對；ROI/event 順序與 linkage
  必須完全相同。結構檢查器依 `visualAttrs_info.matching.<query_id>.evaluated_count` 驗證 production
  policy：至少一個由 PLIP 或 CONCH 提供的 informative attribute → `selected`／
  `informative_visual_attributes_forwarded`，否則為 `skipped`／`insufficient_visual_evidence`。
  Consensus score、Must audit 與個別相似度都不參與 production selection。
