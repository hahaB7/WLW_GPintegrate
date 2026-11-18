# Person C · configuration

Two configs, both read by the pipeline by name (`pipeline/run_pipeline.py`):

| File | `mode` | Use |
|---|---|---|
| `region_proposal.json` | `region_proposal` | The SEGNET student. The pipeline's **production** profile requires exactly this name and mode. |
| `example.json` | `example` | Opens no slide and makes no ROI. The pipeline's **example** profile and the contract tests. |

**Paths** in a config are relative to `components/person_c/`. The checkpoint must resolve below the sibling `reference/person_c/`,
every output directory below the sibling `run/output/`. Do not create run-specific configs in this directory; snapshot the config,
the checkpoint revision and the WSI references with each experiment.

---

## The shipped `region_proposal.json`

```json
{
  "component_version": "person-c/interest-pattern:0.5.0",
  "mode": "region_proposal",
  "segnet": {
    "checkpoint": {
      "path": "../reference/person_c/checkpoint/segnet/c9_sclip_bracs_s0.pt",
      "revision": "sha256:b13895d5cf7ca919cb6b78e43931a5e9174f7305573964c3ed1ac5cea5b3e95b"
    },
    "roi_classes": ["TUM", "BEN"]
  },
  "postprocess_mpp": 1.0,
  "min_object_size": 6400,
  "figures": {
    "dir": "../run/output/materialized/person_c/figures",
    "tissue": false,
    "epithelium": false,
    "boxes": false
  }
}
```

`example.json` is just `component_version` and `mode`.

---

## Keys

### Both modes

| Key | Type | Meaning |
|---|---|---|
| `component_version` | nonempty string | Producer recorded in E and in every selection event. |
| `mode` | `region_proposal` \| `example` | See the table above. |

### `region_proposal`

| Key | Type / range | Meaning |
|---|---|---|
| `segnet.checkpoint` | `{path, revision}` | The student checkpoint. `revision` is `sha256:<64 hex>` and is checked before loading. One checkpoint, one network. |
| `segnet.roi_classes` | nonempty list of strings | Classes whose union is the epithelium mask; names from the checkpoint (`TUM`, `BEN`). |
| `postprocess_mpp` | positive number | MPP of the segmentation and of the mask the boxes are made from (1 µm/px). The figures have the same resolution. |
| `min_object_size` | positive integer | An object with fewer mask pixels (at `postprocess_mpp`) gets no box. 6400 is 6400 µm² at 1 µm/px. |

### `figures` — one JPEG per WSI and stage

Everything under `figures` is optional; all switches are off by default.

| Key | Type | Meaning |
|---|---|---|
| `figures.tissue` | boolean | The class map: one colour per class, a black line between classes, a legend. |
| `figures.epithelium` | boolean | The epithelium mask the boxes are made from (`segnet.roi_classes`, holes filled), tinted, with a black outline. |
| `figures.boxes` | boolean | The ROI boxes (amber) over the epithelium; the caption gives their number. |
| `figures.dir` | path below `run/output` | Directory of the JPEGs, `<name>_<stage>.jpg` (`name`: the stem of the slide's file name). Required when a switch is on. |
| `figures.annotations` | path below `run/` | JSON `{stain_id: [[x, y, w, h], …]}` of annotation ROIs in level-0 pixels. A stain with an entry gets them in lime, numbered, on the boxes figure; a stain without gets nothing. |
| `figures.downscale` | integer 1 … 64 (default 1) | The figure is this many times smaller than the 1 µm/px stages: `4` is 4 µm/px with 16 times fewer pixels, a size most viewers open. The segmentation and the boxes do not change. |

Colours, line width (2 px) and JPEG quality (90) are fixed. How the figures are read: [README, section 4](../README.md#4-figures).

---

## Fixed, not configurable

These are constants of the program, not settings.

| What | Value | Where |
|---|---|---|
| Stain that is processed | H&E (every other stain keeps `roi_num = 0`) | `interest_pattern.py` |
| Device | one CUDA GPU (none → error) | `segmentation.py` |
| Tile and context | 1792 µm and 112 µm | `segmentation.py` |
| Windows per forward pass | 32 | `segmentation.py` |
| Tile-reading processes | 6 | `segmentation.py` |
| Working MPP of `main_info` | from the box area: up to 1, 4, 16, 64, 256, 1024 million px² → 0.25, 0.5, 1, 2, 4, 8 µm/px; larger → 16 | `interest_pattern.py` |
