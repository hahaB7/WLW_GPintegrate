# Person C configuration

`default.json` runs the MRCPS extractor. `model.path` must resolve below sibling
`reference/person_c/checkpoint/`; `roi_output_dir` must resolve below sibling `run/`. Do not create
run-specific configs outside this directory.

| Key | Type / range | Meaning |
|---|---|---|
| `component_version` | nonempty string | Producer recorded in E and each selection event. |
| `backend` | `mrcps`, `segnet` or `demo` | `mrcps` / `segnet` open selected WSI and run inference (`segnet`: the SEGNET student, `configs/segnet.json`). `demo` uses fixed fixture boxes. |
| `stain_type_whitelist` | nonempty list[string] | Case-insensitive stain types allowed to open and process; default is `["HE"]`. |
| `model.path` | nonempty path | Read-only MRCPS checkpoint. Required in `mrcps` mode. |
| `model.revision` | `sha256:<64 hex>` | Optional runtime checksum check; the default pins the approved checkpoint. |
| `device` | `auto`, `cpu`, `cuda` | `auto` uses CUDA when available. Explicit `cuda` fails if unavailable. |
| `patch_size` | positive integer | MRCPS input width and height in read-level pixels; trained value is 512. |
| `stride` | positive integer, at most `patch_size` | Patch grid spacing; trained value is 512. |
| `low_resolution_ratio` | positive integer | Low-resolution context ratio; trained value is 8. |
| `batch_size` | positive integer | Number of patches per model forward pass. Reduce after CUDA OOM. |
| `check_mpp` | positive number | Approximate MPP for tissue background screening. |
| `postprocess_mpp` | positive number | MPP of the stitched mask and morphology. |
| `background_lower`, `background_upper` | integers with `0 <= lower < upper <= 255` | Mean-RGB window used for tissue screening. |
| `background_dilation_kernel` | positive integer | Square grid dilation kernel for background screening. |
| `hole_area_threshold` | nonnegative integer | Fill smaller holes in the stitched mask. |
| `min_object_size` | positive integer | Remove smaller connected tissue objects. |
| `save_roi_png` | boolean | Whether to write PNG crops and set `main_info.roi_path`. |
| `roi_output_dir` | path below sibling `run` | PNG directory when `save_roi_png=true`. |

The legacy area-to-working-MPP ladder is 0.25, 0.5, 1, 2, 4, 8 and 16 µm/pixel for level-0 ROI areas up to 1, 4, 16, 64, 256, 1024 and over 1024 million pixels² respectively. Input and model revisions, resolved config, and WSI references should be snapshotted with each experiment. `demo` output is only a compatibility fixture and does not locate tissue.

`segnet` backend (`configs/segnet.json`; the post-processing keys above `postprocess_mpp`, `hole_area_threshold`, `min_object_size` apply too; `background_*`, `patch_size`, `stride`, `low_resolution_ratio`, `check_mpp` and `model` are MRCPS only):

| Key | Type / range | Meaning |
|---|---|---|
| `segnet.checkpoints` | list of `{path, revision}` | Student checkpoints below `reference/person_c/`; `revision` is `sha256:<64 hex>` and is checked. The probabilities of all are averaged. |
| `segnet.roi_classes` | nonempty list[string] | Classes whose union is the mask (names in the checkpoint; `TUM`, `BEN`). |
| `segnet.tile_um`, `segnet.margin_um` | multiples of 4 | Tile of the slide read at the student's 1 um/px, and its context on every side. |
| `segnet.tta_views` | 1..8 | Dihedral views per window (SEGNET's own default is 8). |
| `segnet.batch_size` | positive integer | Windows per forward pass. |
| `segnet.min_tissue_frac` | 0..1 | A tile with a smaller share of coloured pixels is glass: no ROI. |
| `segnet.nested_overlap` | (0, 1] | After the mask is cut into boxes, a box with at least this share of its area inside a larger kept box is dropped (an object inside another box is not counted again). |
