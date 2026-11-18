# Person C configuration

| File | `mode` | Use |
|---|---|---|
| `region_proposal.json` | `region_proposal` | the SEGNET student; the pipeline's production profile reads exactly this name and mode |
| `example.json` | `example` | no slide is opened; the pipeline's example profile and the contract tests |

Paths in a config are relative to `components/person_c/`: the checkpoint must resolve below sibling `reference/person_c/`,
every output directory below sibling `run/output/`. Do not create run-specific configs outside this directory.
Snapshot the config, the checkpoint revision and the WSI references with each experiment.

## `region_proposal` and `example`

| Key | Type / range | Meaning |
|---|---|---|
| `component_version` | nonempty string | Producer recorded in E and in each selection event. |
| `mode` | `region_proposal`, `example` | See above. |
| `postprocess_mpp` | positive number | MPP of the segmentation and the mask the boxes are made from (1 µm/px). The figures have the same resolution. |
| `min_object_size` | positive integer | Objects with fewer mask pixels (at `postprocess_mpp`) get no box; 6400 is 6400 µm² at 1 µm/px. |

The working MPP of `main_info` follows from the box area in level-0 pixels²: up to 1, 4, 16, 64, 256, 1024 million
0.25, 0.5, 1, 2, 4, 8 µm/px, larger 16.

## `region_proposal` only

| Key | Type / range | Meaning |
|---|---|---|
| `segnet.checkpoint` | `{path, revision}` | The student checkpoint; `revision` is `sha256:<64 hex>` and is checked before loading. One checkpoint, one network. |
| `segnet.roi_classes` | nonempty list[string] | Classes whose union is the epithelium mask; names from the checkpoint (`TUM`, `BEN`). |

### `figures` (`region_proposal` only; one JPEG per WSI and stage, at `postprocess_mpp`)

| Key | Type | Meaning |
|---|---|---|
| `figures.tissue` | boolean | The multi-class tissue segmentation: one colour per class, a legend. |
| `figures.epithelium` | boolean | The epithelium mask the boxes are made from (`segnet.roi_classes`, holes filled). |
| `figures.boxes` | boolean | The level-0 boxes of the ROIs over the epithelium; the title gives their number. |
| `figures.dir` | path below `run/output` | Directory of the JPEGs, `<name>_<stage>.jpg` (`name`: the stem of the slide's file name); required when a switch is on. |
| `figures.annotations` | path below `run/` (optional) | JSON `{stain_id: [[x, y, w, h], ...]}` of annotation ROIs in level-0 pixels. A stain with an entry gets them in lime, numbered, on the boxes figure; a stain without gets nothing. |

Look of the figures (optional; the values shown are the defaults; the colours are fixed):

| Key | Type / range | Meaning |
|---|---|---|
| `figures.downscale` | integer 1..64 (1) | The figure is this many times smaller than the 1 µm/px stages (4: 4 µm/px, 16 times fewer pixels, a viewable overview). |

All switches are off by default.

## `example` only

`component_version` and `mode` are enough: no slide is opened and no ROI is written.
