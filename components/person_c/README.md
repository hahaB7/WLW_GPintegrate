# Person C: WSI interest pattern extraction

Person C reads one raw CaseList case and writes `E.ROIs@2.0`: for every H&E whole-slide image of the case, the
rectangles (ROIs) of the epithelium. Person D (`visual_filter`) reopens the slide with `filepath` and
`level0_info.xywh` to rebuild the pixels; Person E (`clee`) reads the ROI geometry through H.

| | |
|---|---|
| Owner | Person C |
| Component version | `0.5.0` (producer string `person-c/interest-pattern:0.5.0`) |
| Input / output | `CaseListInput@1.0`, one case per call / `E.ROIs@2.0` |
| Entry point | `python -m components.person_c.interest_pattern --input ... --output ... --config ...` |
| Production config | `configs/region_proposal.json` (mode `region_proposal`; the name the pipeline reads) |

## 1. Modes

`mode` in the config selects what the component does. The pipeline reads it: `pipeline/run_pipeline.py` loads
`configs/region_proposal.json` in its production profile (and requires `mode == "region_proposal"`) and `configs/example.json`
in its example profile.

| Config | `mode` | What it does |
|---|---|---|
| `configs/region_proposal.json` | `region_proposal` | Finds the regions of the epithelium with the SEGNET student (section 2). The mode names the stage (region proposal), not the algorithm. |
| `configs/example.json` | `example` | No slide is opened and no ROI is made. For the pipeline's example profile and the contract tests. |

Everything after the mask (boxes, ROI records, E) is shared. `stain_type_whitelist` (default `["HE"]`) selects the stains that are
opened. Every other stain stays in E with `roi_num = 0` and `roi_list = []`.

## 2. Mode `region_proposal`

```text
 WSI ─> tiles of 1792 um at 1 um/px (+112 um context), read by worker processes
     ─> student: 224 px windows at half overlap, cosine-squared blending            GPU, one network
     ─> one probability vector per 4 um cell, enlarged bilinearly to 1 um/px        GPU
     ─> class of the highest probability per pixel                                  GPU      STAGE 1  tissue segmentation (c9)
     ─> union of segnet.roi_classes (TUM, BEN), every hole filled                   GPU / CPU STAGE 2  epithelium
     ─> connected objects, minimum bounding box of each, enclosed boxes dropped     CPU      STAGE 3  boxes
     ─> ROI records ─> E.ROIs
```

- **Model.** DeepLab v3+ (ResNet-101, ASPP, decoder to stride 4), 10 classes (`TUM BEN STR LYM NEC FAT VES BLD SEC OTH`),
  trained in PIPELINE/SEGNET (`Unsup`, spec `c9_224px_1um_sclipcrf_glvq5r2m20-bracs_wsi`, seed 0, step 6000) on unlabeled
  BRACS slides with the pseudo labels of the SCLIP teacher. No annotation was used to train it.
- **No foreground step.** The student labels the glass itself (class OTH), so no tissue detector runs before it.
- **1 µm/px.** The network outputs one vector per 4 µm cell. The probabilities (not the labels) are enlarged bilinearly to
  `postprocess_mpp` (1 µm/px) before the argmax, so the borders are smooth, and everything after works at that resolution.
- **Holes.** The student draws the wall of a gland. A ring that is closed is filled and what lies inside belongs to the ring (no
  box of its own). A ring that is not closed cannot be filled; there a box that lies completely inside a larger box is dropped.
  Boxes that only overlap are all kept.
- **Checkpoint.** `reference/person_c/checkpoint/segnet/c9_sclip_bracs_s0.pt`, pinned by `revision` (SHA-256) in the config and
  checked before it is loaded.
- **What produced the ROIs** is written to `payload.reference_versions.interest_pattern` of E: the mode, the checkpoint with
  its SHA-256, the classes, the tile size, `postprocess_mpp` and `min_object_size`.

## 3. Figures

Optional pictures of the stages, for whoever checks the segmentation. They are drawn after the boxes are final, so they never
change the ROIs. All are off by default; every key is in [configs/README.md](configs/README.md).

| Figure | Switch | Shows |
|---|---|---|
| `tissue` | `figures.tissue` | the class map, one colour per class, with a legend |
| `epithelium` | `figures.epithelium` | the epithelium mask the boxes come from, with its outline |
| `boxes` | `figures.boxes` | the ROI boxes (red, dashed) over the epithelium; the caption gives their number |

- **File.** One JPEG per WSI and stage, `<figures.dir>/<name>_<stage>.jpg`, `name` being the stem of the slide's file name (`BRACS_1228_boxes.jpg`), below `run/output`. A file appears
  whole or not at all.
- **Size.** By default the figure has the resolution of the segmentation, 1 µm/px: a 60 000 × 70 000 px slide gives about
  16 000 × 18 000 px (about 50 MB). `figures.downscale` makes the figure that many times smaller (`4`: 4 µm/px, 16 times fewer
  pixels). It only shrinks the picture; the segmentation and the boxes are the same. `figures.quality` is the JPEG quality and
  `figures.box_thickness` the line width (default 2 px; the figure is meant to be zoomed into). Colours are fixed.
- **Annotations.** If `figures.annotations` points to a JSON `{stain_id: [[x, y, w, h], ...]}` (level-0 pixels), the annotation
  ROIs of that stain are drawn in green on the boxes figure and counted in its caption. Each has a dashed box, a green ring around
  it and a number, so that it is found at a glance in a zoomed-out figure. Without an entry nothing is drawn.

**Looking at a figure.** A full-size figure is too large for most viewers and for a web page. Either set `figures.downscale`, or
copy the file to a computer (`scp`) and open it with a viewer for large images (Fiji / ImageJ, IrfanView, an image editor), or
cut an overview and crops where it is:

```python
import cv2
f = "BRACS_1228_boxes.jpg"
cv2.imwrite("overview.jpg", cv2.imread(f, cv2.IMREAD_REDUCED_COLOR_8))     # 1/8 of the size: 8 um/px, about 2000 px
cv2.imwrite("crop.jpg", cv2.imread(f)[6500:8300, 3800:5800])               # a 2000 x 1800 px window at 1 um/px (rows, columns)
```

Decoding a whole figure needs several GB of memory: do it inside a job, not on a login node.

## 4. Coordinates and ROI rules (downstream relies on this section)

- **Level 0** is the full-resolution image, origin at the upper left, x to the right, y down. `level0_info.xywh` is
  `[x, y, width, height]` in level-0 pixels and covers `[x, x + width) × [y, y + height)`; it always lies inside the slide.
  `level0_info.mpp` is `[mpp_x, mpp_y]` read from the slide (`openslide.mpp-x/y`); a slide without MPP fails.
- **From mask to box.** The epithelium mask is at `postprocess_mpp` (1 µm/px). Its connected objects (4-connectivity) with at
  least `min_object_size` pixels each get one box: the minimum bounding box, with the lower edge floored and the upper edge
  ceiled after scaling to level 0, clamped to the slide. Nothing is dilated. A box that lies completely inside another box is
  dropped.
- **`main_info`** is the same box in a virtual image resampled to a working MPP chosen from the box area in level-0 pixels²
  (up to 1, 4, 16, 64, 256, 1024 million: 0.25, 0.5, 1, 2, 4, 8 µm/px; larger: 16). `xywh` is floored / ceiled by the same
  scaling and clamped; `mpp` is the scalar working MPP. `cxcywh` of both records is
  `[cx / W, cy / H, w / W, h / H]` of the image of that record (contract clarification CCR-C-001).
  `area` is `width × height` in the pixels of that record.
- **Identity.** `roi_id` is `<case_id>-<stain_id>-roi-<16 hex>`, the hash of the case, the stain and the level-0 box.
  `global_idx` counts the ROIs of the case in stain order, `local_idx` those of the stain. The ROIs of a stain are ordered by
  (top, left).
- **Selection history.** One event per ROI: stage `interest_pattern_extraction`, owner `person_C`, action
  `candidate_generated`, status `selected`. `DxPair`, `visualAttrs` and `visualAttrs_info` are `null`.
- **Empty.** A stain without objects gives `roi_num = 0`; a case without any ROI is a valid E.
- **`save_roi_png`**: `true` writes the crop of every ROI below `run/output` and sets `main_info.roi_path`; otherwise downstream
  components reopen the slide.
- **Failure.** A case with more than one entry, a missing slide, a missing MPP, a missing or wrong checkpoint, an unavailable
  CUDA device that was asked for, a model error, or an E that violates the contract ends with a non-zero exit code and
  writes no E. (A figure that was written before a later failure may remain in `figures.dir`.)

## 5. Code structure

```text
components/person_c/
├── interest_pattern.py       entry point: CLI, the stain loop, E.ROIs (contract boundary)
├── wsi_processing.py         settings check, slide MPP and levels, mask -> boxes, ROI geometry, no torch
├── region/
│   ├── segnet.py             the student on a slide: checkpoint, windows, tiles, the stages (torch, openslide)
│   ├── segnet_net.py         DeepLab v3+ network, inference copy of PIPELINE/SEGNET's SegNet
│   └── figures.py            the JPEG of a stage (torch for the painting, OpenCV for text and writing)
├── configs/                  region_proposal.json, example.json, README.md (every key)
├── examples/                 canonical CaseList and expected E (example mode), command, e2e/run_e2e.sh
├── tests/                    test_interest_pattern.py (contract, E), test_segnet_regions.py (settings, boxes, figures)
├── component.yaml  external-assets.yaml  requirements.txt  Dockerfile  PREPARATION.md
```

The call chain of one case:

```text
interest_pattern.main
  └─ run(source, config)
       ├─ wsi.validate_settings(config)                    every key checked here, nowhere else
       ├─ segnet.load_model(settings)                      checkpoint path + SHA-256, classes, device
       └─ per stain (whitelist), per WSI:
            ├─ wsi.slide_mpp(slide)
            ├─ segnet.infer_stages(path, slide, mpp, ...)  -> {"tissue", "epithelium", "classes", ["image"]}
            ├─ wsi.level0_boxes(epithelium, ...)           -> boxes (connected objects)
            ├─ wsi.drop_enclosed(boxes)
            ├─ _make_roi(...)                              level-0 and main_info records, optional crop
            └─ figures.save_figures(...)                   only if a switch is on
```

Where the work runs:

| Step | Where | Why |
|---|---|---|
| Reading tiles | `segnet.num_workers` CPU processes (`TileReader`, one slide handle each) | decoding the slide is the slowest part; the GPU predicts the previous tile meanwhile |
| Windows, enlarging, argmax, label map, union of the classes | GPU | per-pixel work |
| Hole filling (`cv2.floodFill`), connected objects (`cv2.connectedComponentsWithStats`) | CPU | one pass over the mask, about 1 s; torch has no equivalent |
| Painting of the figures | GPU, in bands of 1024 rows | blending and outline per pixel |
| JPEG encoding | CPU threads, one per figure | |

Rules for changes:

- Every config key is validated in `wsi.validate_settings` (and its helpers) and documented in `configs/README.md`; a new key is
  added to both and to a test.
- `wsi_processing.py` imports no torch, so the contract tests need no GPU. Heavy imports (torch, openslide, cv2) are inside the
  functions of `region/`.
- The ROI geometry (section 4) is what Person D and E rely on; a change to it is a contract question, not a code cleanup.
- The network code in `segnet_net.py` must keep the module names of PIPELINE/SEGNET's `SegNet`: the checkpoint is loaded with
  `strict` key checking.

## 6. Configuration

See [configs/README.md](configs/README.md) for every key.

## 7. Run

```bash
python -m components.person_c.interest_pattern \
  --input  ../run/input/pipeline/case-001.json \
  --output ../run/output/work/E_rois.json \
  --config components/person_c/configs/region_proposal.json
```

`--input` and `--output` must be below the sibling `run/`, `--config` below `components/person_c/configs/`. On a cluster, run it
inside a job (one GPU, 8 CPUs; memory in section 9); a real slide is not a job for a login node.
`examples/e2e/run_e2e.sh` does the same for one slide and prints the ROI summary.

Tests (no GPU, no slide, no checkpoint):

```bash
python -m unittest discover -s components/person_c/tests -t . -v
```

Container (build from the repository root; `reference/` and `run/` are mounted, not copied into the image). The image was
written but **has not been built** (no Docker where this was developed). The pipeline starts `wlw/person-c:0.1.0-lab23-production`
(its own tag, overridable with `WLW_IMAGE_PERSON_C`); build with that tag, or set the variable to the tag built here:

```bash
docker build -f components/person_c/Dockerfile -t wlw/person-c:0.5.0 .
docker run --rm --gpus all \
  -v "$(pwd)/../reference:/reference:ro" -v "$(pwd)/../run:/run" \
  wlw/person-c:0.5.0 \
    --input  /run/input/pipeline/case-001.json \
    --output /run/output/work/E_rois.json \
    --config /app/components/person_c/configs/region_proposal.json
```

## 8. Environment and assets

| | |
|---|---|
| Python | 3.10 in the image (`Dockerfile`); the development runs used 3.12 |
| WSI library | `openslide-python 1.4.6` (any format OpenSlide reads that carries MPP; SVS was used) |
| Framework | `torch 2.6.0`, `torchvision 0.21.0` in the image (cu118 wheels); development runs: torch 2.14.0 with CUDA 13.0 |
| Other | `numpy`, `opencv-python-headless`, `Pillow` (`requirements.txt`); `scipy` only for one test (it checks the hole filling against `scipy.ndimage.binary_fill_holes`) |
| GPU | one; peak GPU memory 1.05 GiB measured during inference (one network, `batch_size` 32) |
| CPU | 8 (6 tile readers and the main process); `device: "cpu"` is accepted but was not run |
| Checkpoint | `reference/person_c/checkpoint/segnet/c9_sclip_bracs_s0.pt`, SHA-256 `b13895d5cf7ca919cb6b78e43931a5e9174f7305573964c3ed1ac5cea5b3e95b`, 238 008 769 bytes |

External assets (the checkpoint, a test slide) are listed with digests in `external-assets.yaml`. Nothing is downloaded at build or
run time; `reference/` is read-only, everything the component writes goes below `run/`. The CONCH checkpoint and
`template_ref/bcss_vocab.json` of the previous region proposal are no longer used.

## 9. Measured performance

One H200, 8 CPUs, `configs/region_proposal.json`, BRACS slides (SVS, 0.2524 µm/px).

| Slide | Run | Time | Peak host memory | ROIs |
|---|---|---|---|---|
| BRACS_1003694 | CLI, no figures | 37 s | 7.3 GB | 40 |
| BRACS_1003691 (61 879 × 70 581 px) | student, label map, hole filling, boxes (model loaded) | 20 s | | 152 |
| BRACS_1003691 | the same, then three figures | 32 s in all | | 152 |
| BRACS_1587 (about 22 000 × 33 000 px at 1 µm) | CLI, three figures | 74 s | 15.7 GB | 5 |
| 9 slides of 0.7 to 2.4 GB | CLI, three figures | 34 to 106 s each | 15.7 GB (largest) | 5 to 203 |

The boxes of BRACS_1003691 take 0.5 s on the CPU (the hole filling is not timed separately; it is part of the 20 s above). The first version of this mode took 6 min 2 s
on BRACS_1003691: the tiles were read at the full-resolution level (a requested 3.96 µm/px is 1 % below the 4.0 of level 1, and
`get_best_level_for_downsample` then skips it) and in the main loop; they are now read at the right level
(`wsi_processing.read_level`, 1 % tolerance) by `segnet.num_workers` processes. The memory of a run with figures is mostly the slide image
and the three figures at 1 µm/px.

## 10. Limits

- **The ROIs are not scored.** The segmentation and the boxes were looked at in the figures of a sample of BRACS slides; no ROI was compared
  with an annotation by a number. The student's own scores are in PIPELINE/SEGNET, not here.
- **The student was trained on BRACS slides** (breast). A BRACS slide is therefore not an independent test, and slides from
  other sites or organs may be segmented worse.
- **A box is axis-aligned.** An elongated or diagonal object gets a box that is mostly empty; overlapping boxes of neighbouring
  objects are kept. A tumour that infiltrates in single cells gives many small boxes. `coords_seg` is `null`.
- **A ring that is not closed is not filled**, so a gland whose wall has a gap keeps its lumen outside the mask.
- **Output is deterministic** for a given checkpoint (an argmax, no sampling); differences between GPU models were not checked.
- **Slide size.** A figure of a very large slide has more than 2^31 bytes of pixels; the figures are written as BGR for that
  reason (a conversion copy of such an image gave black files). Slides above about 1 gigapixel at 1 µm/px were not tried.
- **Not run:** the container image, the CPU mode, the full pipeline (the reference assets of Person A and E were not at hand),
  and Person D / E reading these ROIs.

## 11. Notes for the pipeline

Nothing outside `components/person_c/` was changed. What the pipeline owner may want to know:

- `mode: "region_proposal"` names the stage. In this version the algorithm behind it is SEGNET, not the CONCH proposal that
  PR #1 delivered. The CONCH code and checkpoint are not needed any more; `pipeline/run_pipeline.py` needs no change.
- `reference/person_c/checkpoint/segnet/c9_sclip_bracs_s0.pt` must exist in the mounted `reference/` (see `external-assets.yaml`).
- The pipeline's image tag for C is `wlw/person-c:0.1.0-lab23-production`; this component is `0.5.0`.
- `payload.reference_versions.interest_pattern` of E changed its content (it recorded the CONCH region settings before).
- The walk-through in `components/person_d/examples/person_c_integration/README.md` calls `build_region_artifact`, which is
  PR #1's code (commit `0ad461d`), not this version.
- The ROIs differ from the earlier ones in number and shape (epithelium objects, not CONCH regions).

## 12. Changelog

| Version | Change |
|---|---|
| 0.5.0 | The SEGNET student replaces the CONCH region proposal (and the MRCPS segmentation before it; both stay in git history). `mode` replaces `backend`; `configs/region_proposal.json` and `configs/example.json` are the names the pipeline reads. 1 µm/px segmentation (bilinear enlargement before the argmax), GPU per-pixel work, tiles read by worker processes, JPEG figures of the stages and optional annotation ROIs. E records what produced the ROIs in `reference_versions`. |
| 0.4.0 | MRCPS extractor and example mode (upstream). |
