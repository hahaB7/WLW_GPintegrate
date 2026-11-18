# Person C configuration

| File | Backend | Use |
|---|---|---|
| `default.json` | `segnet` | the SEGNET student (the default of the pipeline) |
| `mrcps.json` | `mrcps` | the MRCPS network, kept as the comparison |
| `demo.json` | `demo` | fixed boxes, no slide; pipeline and contract tests only |

Paths in a config are relative to `components/person_c/`: checkpoints must resolve below sibling `reference/person_c/`,
every output directory below sibling `run/output/`. Do not create run-specific configs outside this directory.
Snapshot the config, the checkpoint revision and the WSI references with each experiment.

## Every backend that opens a slide (`segnet`, `mrcps`)

| Key | Type / range | Meaning |
|---|---|---|
| `component_version` | nonempty string | Producer recorded in E and in each selection event. |
| `backend` | `segnet`, `mrcps`, `demo` | See above. |
| `stain_type_whitelist` | nonempty list[string] | Case-insensitive stain types that are opened; the others keep `roi_list = []`. Default `["HE"]`. |
| `device` | `auto`, `cpu`, `cuda` | `auto` uses CUDA when available; an explicit `cuda` fails when it is not. |
| `postprocess_mpp` | positive number | MPP of the mask the boxes are made from (8 µm/px). |
| `hole_area_threshold` | nonnegative integer | Holes of at most this many mask pixels are filled before the boxes are made. |
| `min_object_size` | positive integer | Objects smaller than this many mask pixels get no box. |
| `save_roi_png` | boolean | Write the crop of every ROI and set `main_info.roi_path`. |
| `roi_output_dir` | path below `run/output` | Where the crops go; required when `save_roi_png` is true. |

The working MPP of `main_info` follows from the box area in level-0 pixels²: up to 1, 4, 16, 64, 256, 1024 million
0.25, 0.5, 1, 2, 4, 8 µm/px, larger 16.

## `segnet` only

| Key | Type / range | Meaning |
|---|---|---|
| `segnet.checkpoints` | list of `{path, revision}` | Student checkpoints; `revision` is `sha256:<64 hex>` and is checked before loading. The probabilities of all are averaged; every extra network costs as much GPU time as the first (default: one). |
| `segnet.roi_classes` | nonempty list[string] | Classes whose union is the epithelium mask; names from the checkpoint (`TUM`, `BEN`). |
| `segnet.tile_um`, `segnet.margin_um` | multiples of 4 | The slide is read in tiles of `tile_um` at the student's 1 µm/px, with `margin_um` of context on every side. |
| `segnet.tta_views` | 1..8 | Dihedral views per window. 1 is no test-time augmentation (default); SEGNET's own default is 8, and every view multiplies the GPU time. |
| `segnet.batch_size` | positive integer | Windows per forward pass. Reduce after a CUDA out-of-memory error. |
| `segnet.num_workers` | integer >= 0 | Processes that read the tiles while the GPU predicts (0: the main process reads). At most the CPUs of the job minus two. |

### `figures` (`segnet` only; one PNG per WSI and stage)

| Key | Type | Meaning |
|---|---|---|
| `figures.tissue` | boolean | The multi-class tissue segmentation: one colour per class, a legend. |
| `figures.epithelium` | boolean | The epithelium mask the boxes are made from (`segnet.roi_classes`, holes filled). |
| `figures.boxes` | boolean | The level-0 boxes of the ROIs over the epithelium; the title gives their number. |
| `figures.dir` | path below `run/output` | Directory of the PNGs, `<case_id>_<stain_id>_<stage>.png`; required when a switch is on. |

All switches are off by default.

## `mrcps` only

| Key | Type / range | Meaning |
|---|---|---|
| `model.path` | path below `reference/person_c/checkpoint/` | Read-only MRCPS checkpoint. |
| `model.revision` | `sha256:<64 hex>` | Optional checksum; the default pins the approved checkpoint. |
| `patch_size`, `stride` | positive integers, `stride <= patch_size` | Input size and grid spacing in read-level pixels; trained value 512. |
| `low_resolution_ratio` | positive integer | Ratio of the low-resolution context; trained value 8. |
| `batch_size` | positive integer | Patches per forward pass. |
| `check_mpp` | positive number | Approximate MPP of the overview that screens the background. |
| `background_lower`, `background_upper` | integers, `0 <= lower < upper <= 255` | Mean-RGB window of a tissue patch. |
| `background_dilation_kernel` | positive integer | Square kernel that dilates the screened grid. |

## `demo` only

`magnification`, `demo_mpp` (`x`, `y`) and `demo_regions` (a list of `x`, `y`, `width`, `height`, `level`): the fixed boxes.
