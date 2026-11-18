# Person C configuration

| File | `mode` | Use |
|---|---|---|
| `region_proposal.json` | `region_proposal` | the SEGNET student; the pipeline's production profile reads exactly this name and mode |
| `example.json` | `example` | no slide is opened; the pipeline's example profile and the contract tests |

Paths in a config are relative to `components/person_c/`: checkpoints must resolve below sibling `reference/person_c/`,
every output directory below sibling `run/output/`. Do not create run-specific configs outside this directory.
Snapshot the config, the checkpoint revision and the WSI references with each experiment.

## `region_proposal` and `example`

| Key | Type / range | Meaning |
|---|---|---|
| `component_version` | nonempty string | Producer recorded in E and in each selection event. |
| `mode` | `region_proposal`, `example` | See above. |
| `stain_type_whitelist` | nonempty list[string] | Case-insensitive stain types that are opened; the others keep `roi_list = []`. Default `["HE"]`. |
| `device` | `auto`, `cpu`, `cuda` | `auto` uses CUDA when available; an explicit `cuda` fails when it is not. |
| `postprocess_mpp` | positive number | MPP of the segmentation and the mask the boxes are made from (1 µm/px). The figures have the same resolution. |
| `min_object_size` | positive integer | Objects with fewer mask pixels (at `postprocess_mpp`) get no box; 6400 is 6400 µm² at 1 µm/px. |
| `save_roi_png` | boolean | Write the crop of every ROI and set `main_info.roi_path`. |
| `roi_output_dir` | path below `run/output` | Where the crops go; required when `save_roi_png` is true. |

The working MPP of `main_info` follows from the box area in level-0 pixels²: up to 1, 4, 16, 64, 256, 1024 million
0.25, 0.5, 1, 2, 4, 8 µm/px, larger 16.

## `region_proposal` only

| Key | Type / range | Meaning |
|---|---|---|
| `segnet.checkpoints` | list of `{path, revision}` | Student checkpoints; `revision` is `sha256:<64 hex>` and is checked before loading. The probabilities of all are averaged; every extra network costs as much GPU time as the first (default: one). |
| `segnet.roi_classes` | nonempty list[string] | Classes whose union is the epithelium mask; names from the checkpoint (`TUM`, `BEN`). |
| `segnet.tile_um`, `segnet.margin_um` | multiples of 4 | The slide is read in tiles of `tile_um` at the student's 1 µm/px, with `margin_um` of context on every side. |
| `segnet.tta_views` | 1..8 | Dihedral views per window. 1 is no test-time augmentation (default); SEGNET's own default is 8, and every view multiplies the GPU time. |
| `segnet.batch_size` | positive integer | Windows per forward pass. Reduce after a CUDA out-of-memory error. |
| `segnet.num_workers` | integer >= 0 | Processes that read the tiles while the GPU predicts (0: the main process reads). At most the CPUs of the job minus two. |

### `figures` (`region_proposal` only; one JPEG per WSI and stage, at `postprocess_mpp`)

| Key | Type | Meaning |
|---|---|---|
| `figures.tissue` | boolean | The multi-class tissue segmentation: one colour per class, a legend. |
| `figures.epithelium` | boolean | The epithelium mask the boxes are made from (`segnet.roi_classes`, holes filled). |
| `figures.boxes` | boolean | The level-0 boxes of the ROIs over the epithelium; the title gives their number. |
| `figures.dir` | path below `run/output` | Directory of the JPEGs, `<case_id>_<stain_id>_<stage>.jpg`; required when a switch is on. |
| `figures.annotations` | path below `run/` (optional) | JSON `{stain_id: [[x, y, w, h], ...]}` of annotation ROIs in level-0 pixels. A stain with an entry gets them in green on the boxes figure; a stain without gets nothing. |

All switches are off by default.

## `example` only

`component_version` and `mode` are enough: no slide is opened and no ROI is written. Optional: `demo_mpp` (`x`, `y`) and `demo_regions` (a list of `x`, `y`, `width`, `height`, `level`) give fixed boxes; `stain_type_whitelist` as above.
