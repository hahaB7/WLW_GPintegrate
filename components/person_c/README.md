# Person C: WSI interest pattern extraction

Person C reads one raw CaseList case and writes `E.ROIs@2.0`: for every H&E whole-slide image of the case, the
rectangles (ROIs) of the epithelium. Person D (`visual_filter`) reopens the slide with `filepath` and
`level0_info.xywh` to rebuild the pixels; Person E (`clee`) reads the ROI geometry through H.

| | |
|---|---|
| Component version | `0.5.0` (producer string `person-c/interest-pattern:0.5.0`) |
| Input / output | `CaseListInput@1.0`, one case per call / `E.ROIs@2.0` |
| Entry point | `python -m components.person_c.interest_pattern --input ... --output ... --config ...` |
| Production config | `configs/region_proposal.json` (mode `region_proposal`) |

## 1. Modes

`mode` in the config selects what the component does; the pipeline reads it (`pipeline/run_pipeline.py` requires
`region_proposal` in its production profile and `configs/region_proposal.json` / `configs/example.json` by name).

| Config | `mode` | What it does |
|---|---|---|
| `configs/region_proposal.json` | `region_proposal` | Finds the regions of the epithelium with the SEGNET student (section 2). The mode names the stage (region proposal), not the algorithm. |
| `configs/example.json` | `example` | No slide is opened: no ROIs, or the fixed boxes of `demo_regions` if the config has them. For the pipeline's `example` profile and the contract tests. |

Everything after the mask (boxes, ROI records, E) is shared. `stain_type_whitelist` (default `["HE"]`) selects the stains that are
opened. Every other stain stays in E with `roi_num = 0` and `roi_list = []`.

## 2. Mode `region_proposal`

```text
 WSI ─> tiles of 1792 um at 1 um/px (+112 um context), read by worker processes
     ─> student: 224 px windows at half overlap, cosine-squared blending         one network, no test-time augmentation
     ─> class of the highest probability per 4 um cell                          STAGE 1  tissue segmentation (c9)
     ─> union of segnet.roi_classes (TUM, BEN), every hole filled               STAGE 2  epithelium
     ─> connected objects, minimum bounding box of each, enclosed boxes dropped STAGE 3  boxes
     ─> ROI records ─> E.ROIs
```

- **Model.** DeepLab v3+ (ResNet-101, ASPP, decoder to stride 4), 10 classes (`TUM BEN STR LYM NEC FAT VES BLD SEC OTH`),
  trained in PIPELINE/SEGNET (`Unsup`, spec `c9_224px_1um_sclipcrf_glvq5r2m20-bracs_wsi`, seed 0, step 6000) on unlabeled
  BRACS slides with the pseudo labels of the SCLIP teacher. No annotation was used to train it. The network code is
  `region/segnet_net.py` (inference copy of SEGNET's `SegNet`); the mask code is `region/segnet.py`.
- **No foreground step.** The student labels the glass itself (class OTH), so no tissue detector runs before it.
- **Holes.** The student draws the wall of a gland; a ring that is closed is filled, and what lies inside belongs to the ring
  (no box of its own). A ring that is not closed cannot be filled; there a box that lies completely inside a larger box is
  dropped. Boxes that only overlap are all kept.
- **Figures.** `figures.tissue`, `figures.epithelium` and `figures.boxes` (default off) write one PNG per WSI and stage to
  `figures.dir` below `run/output`, as `<case_id>_<stain_id>_<stage>.png`: the class map with a legend, the epithelium mask,
  and the boxes over it. A file appears whole or not at all. Code: `region/figures.py`.
- **Checkpoint.** `reference/person_c/checkpoint/segnet/c9_sclip_bracs_s0.pt`, pinned by `revision` (SHA-256) in the config and
  checked before it is loaded.

## 3. Coordinates and ROI rules (downstream relies on this section)

- **Level 0** is the full-resolution image, origin at the upper left, x to the right, y down. `level0_info.xywh` is
  `[x, y, width, height]` in level-0 pixels and covers `[x, x + width) × [y, y + height)`; it always lies inside the slide.
  `level0_info.mpp` is `[mpp_x, mpp_y]` read from the slide (`openslide.mpp-x/y`); a slide without MPP fails.
- **From mask to box.** The mask is at `postprocess_mpp` (8 µm/px). Holes of at most `hole_area_threshold` pixels are filled,
  the mask is dilated by a disk of radius 1, objects smaller than `min_object_size` pixels are removed, and the connected
  objects (4-connectivity) are labelled. The box of an object is its minimum bounding box, with the lower edge floored
  and the upper edge ceiled after scaling to level 0, and clamped to the slide.
- **`main_info`** is the same box in a virtual image resampled to a working MPP chosen from the box area in level-0 pixels²
  (up to 1, 4, 16, 64, 256, 1024 million: 0.25, 0.5, 1, 2, 4, 8 µm/px; larger: 16). `xywh` is floored / ceiled by the same
  scaling and clamped; `mpp` is the scalar working MPP. `cxcywh` of both records is
  `[cx / W, cy / H, w / W, h / H]` of the image of that record (contract clarification CCR-C-001).
  `area` is `width × height` in the pixels of that record.
- **Identity.** `roi_id` is `<case_id>-<stain_id>-roi-<16 hex>`, the hash of the case, the stain and the level-0 box.
  `global_idx` counts the ROIs of the case in stain order, `local_idx` those of the stain. The ROIs of a
  stain are ordered by (top, left).
- **Selection history.** One event per ROI: stage `interest_pattern_extraction`, owner `person_C`, action
  `candidate_generated`, status `selected`. `DxPair`, `visualAttrs` and `visualAttrs_info` are `null`.
- **Empty.** A stain without objects gives `roi_num = 0`; a case without any ROI is a valid E.
- **`save_roi_png`** (`region_proposal`): `true` writes the crop of every ROI below `run/output` and sets
  `main_info.roi_path`; otherwise downstream components reopen the slide.
- **Failure.** A case with more than one entry, a missing slide, a missing MPP, a missing or wrong checkpoint, an
  unavailable CUDA device that was asked for, a model error, or an E that violates the contract ends with a non-zero exit
  code and writes no E. (A figure that was written before a later failure may remain in `figures.dir`.)

## 4. Configuration

See [configs/README.md](configs/README.md) for every key.

## 5. Run

```bash
python -m components.person_c.interest_pattern \
  --input  ../run/input/pipeline/case-001.json \
  --output ../run/output/work/E_rois.json \
  --config components/person_c/configs/region_proposal.json
```

`--input` and `--output` must be below the sibling `run/`, `--config` below `components/person_c/configs/`. On the
cluster, run it inside a Slurm job (one GPU, 8 CPUs, 8 GB of memory are enough); a real slide is not a job for the login node.
`examples/e2e/run_e2e.sh` does the same for one slide and prints the ROI summary.

Tests (no GPU, no slide):

```bash
python -m unittest discover -s components/person_c/tests -t . -v
```

Container (build from the repository root; `reference/` and `run/` are mounted, not copied into the image). The image was
written but **has not been built** (this cluster has no Docker):

```bash
docker build -f components/person_c/Dockerfile -t wlw/person-c:0.5.0 .
docker run --rm --gpus all \
  -v "$(pwd)/../reference:/reference:ro" -v "$(pwd)/../run:/run" \
  wlw/person-c:0.5.0 \
    --input  /run/input/pipeline/case-001.json \
    --output /run/output/work/E_rois.json \
    --config /app/components/person_c/configs/region_proposal.json
```

## 6. Environment and assets

| | |
|---|---|
| Python | 3.10 in the image (`Dockerfile`); the development runs used 3.12 |
| WSI library | `openslide-python 1.4.6` (any format that OpenSlide reads and that carries MPP; SVS was used) |
| Framework | `torch 2.6.0`, `torchvision 0.21.0` in the image (cu118 wheels); development runs: torch 2.14.0 with CUDA 13.0 |
| Other | `numpy`, `opencv-python-headless`, `scikit-image`, `scipy`, `Pillow` (`requirements.txt`) |
| GPU | one; peak GPU memory 0.78 GiB measured (one network, no TTA, `batch_size` 32) |
| CPU | 8 (6 tile readers and the main process); `device: "cpu"` is accepted but was not run |
| Checkpoint | `reference/person_c/checkpoint/segnet/c9_sclip_bracs_s0.pt`, SHA-256 `b13895d5cf7ca919cb6b78e43931a5e9174f7305573964c3ed1ac5cea5b3e95b`, 238 008 769 bytes |

External assets (checkpoints, test slide) are listed with digests in `external-assets.yaml`. Nothing is downloaded at
build or run time; `reference/` is read-only, everything the component writes goes below `run/`.

## 7. Measured performance

One slide only: BRACS_1003691 (61 879 × 70 581 px, 0.2524 µm/px), one H200, 8 CPUs, `configs/region_proposal.json`.

| | |
|---|---|
| Whole CLI run, three PNGs on | 27.7 s |
| Student and boxes (model already loaded) | 20.3 s |
| Peak GPU memory / peak host memory | 0.78 GiB / 2.4 GB |
| ROIs | 197 |

The first version took 6 min 2 s on the same slide: the tiles were read at the full-resolution level
(a requested 3.96 µm/px is 1 % below the 4.0 of level 1, and `get_best_level_for_downsample` then skips it) and in the main
loop. They are now read at the right level (`wsi_processing.read_level`, 1 % tolerance) by `segnet.num_workers` processes.
Test-time augmentation (`segnet.tta_views`) and more checkpoints multiply the GPU time and are off / one by default.

## 8. Limits

- **The ROIs are not scored.** The segmentation was inspected on one slide (the figures); no ROI was compared with an
  annotation. The student's own scores are in PIPELINE/SEGNET, not here.
- **The student was trained on BRACS slides** (breast). A BRACS slide is therefore not an independent test, and slides from
  other sites or organs may be segmented worse.
- **A box is axis-aligned.** An elongated or diagonal object gets a box that is mostly empty; overlapping boxes of
  neighbouring objects are kept. `coords_seg` is `null`.
- **Output is deterministic** for a given checkpoint (an argmax, no sampling); differences between GPU models were not checked.
- **Only one slide** was timed, and the container image was not built.
- **Downstream round trip** (Person D / E reading these ROIs) was not tested with their real implementations.

## 9. Files

```text
components/person_c/
├── interest_pattern.py       entry point: CLI, stain loop, E.ROIs
├── wsi_processing.py         settings check, MPP and levels, mask to boxes, ROI records
├── region/
│   ├── segnet.py             the student on a slide: tiles, windows, class map, epithelium mask
│   ├── segnet_net.py         DeepLab v3+ network (inference)
│   └── figures.py            the PNG of a stage
├── configs/                  region_proposal.json, example.json, README.md
├── examples/                 canonical CaseList and expected E, command, e2e/run_e2e.sh
├── tests/                    test_interest_pattern.py, test_segnet_regions.py
├── component.yaml  external-assets.yaml  requirements.txt  Dockerfile
└── PREPARATION.md  CONTRACT_CHANGE_REQUESTS.md  DELIVERY_REPORT.md   (hand-over notes)
```

## 10. Changelog

| Version | Change |
|---|---|
| 0.5.0 | The SEGNET student replaces the MRCPS segmentation (the MRCPS code, network and config were removed; they stay in git history). `configs/region_proposal.json` is the config the pipeline reads (its production profile requires that name and `mode: "region_proposal"`). Stage PNGs (`figures`). Tiles read at the right level by worker processes (6 min to 28 s). The unreachable CONCH region-proposal code was removed. |
| 0.4.0 | MRCPS extractor and example mode (upstream). |
