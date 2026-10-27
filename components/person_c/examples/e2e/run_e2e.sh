#!/usr/bin/env bash
# Real WSI -> CONCH -> ROI run of Person C (needs a GPU, the assets under ../reference/person_c
# and one slide with MPP metadata). Usage, from the repository root:
#   SLIDE=/path/to/slide.svs PYTHON=python bash components/person_c/examples/e2e/run_e2e.sh
# Writes ../run/output/work/E_rois_e2e.json and prints the ROI summary. Any failure exits non-zero.
set -euo pipefail
: "${SLIDE:?set SLIDE to the slide path}"
PYTHON="${PYTHON:-python}"
CASE_SRC=components/person_c/examples/e2e/case_list.bracs_1003691.json
CASE=../run/input/pipeline/bracs-1003691.json
OUT=../run/output/work/E_rois_e2e.json
mkdir -p ../run/input/pipeline ../run/output/work
"$PYTHON" - "$CASE_SRC" "$CASE" "$SLIDE" <<'PY'
import json, sys
src, dst, slide = sys.argv[1:]
data = json.load(open(src))
data["case_list"][0]["tissue_blocks"][0]["stains"][0]["filepath"] = slide
json.dump(data, open(dst, "w"), indent=2)
PY
rm -f "$OUT"
"$PYTHON" -m components.person_c.interest_pattern --input "$CASE" --output "$OUT" \
  --config components/person_c/configs/region_proposal.json
"$PYTHON" - "$OUT" <<'PY'
import json, sys
art = json.load(open(sys.argv[1]))
stain = art["payload"]["case_list"][0]["tissue_blocks"][0]["stains"][0]
sides = sorted(max(r["level0_info"]["roi_wh"]) for r in stain["roi_list"])
print(f"contract={art['contract']}@{art['schema_version']} rois={stain['roi_num']} "
      f"level0_longest_side_px min={sides[0] if sides else None} max={sides[-1] if sides else None}")
PY
