# Person C canonical example

A contract-level example that needs no GPU, slide or model. It feeds fixed proposal polygons through
the real `build_region_artifact` (polygon -> ROI) path and compares the result with a frozen output.

| File | Content |
|---|---|
| `case_list.valid.json` | one case, two stains: `HE` (processed) and `IHC` (not processed, `stain_types=["HE"]`) |
| `proposals.valid.json` | GeoJSON polygons standing in for the CONCH region-proposal output (level-0 px) |
| `E_rois.expected.json` | expected `E.ROIs@2.0` for slide 30000 x 20000 px, MPP 0.25, config below |

Slide and config used by `tests/test_interest_pattern.py::CanonicalExampleTests`
(`EXAMPLE_SLIDE`, `EXAMPLE_CONFIG`): `target_mpp=0.5`, `min_region_area_px=262144`,
`max_rois_per_stain=256`, `max_main_side_px=2048`.

Covered cases (5 polygons, 4 ROIs):

1. ordinary component with fractional edges (rounded outwards);
2. component with a hole (the box ignores the hole);
3. tiny component (area < `min_region_area_px`): dropped;
4. component sticking out of the slide: box clamped to the slide;
5. very large component: `level0_info` keeps the full box, `main_info.mpp` grows so the longest
   side is 2048 px;
6. non-processed stain (`IHC`): kept with `roi_num = 0`, `roi_list = []`.

Run: `python -m unittest components.person_c.tests.test_interest_pattern`

## End-to-end example (real WSI -> CONCH -> ROI)

`e2e/case_list.bracs_1003691.json` is a one-stain CaseList for the BRACS test slide
`BRACS_1003691.svs` (61879 x 70581 px, MPP 0.2524; not redistributed here). `e2e/run_e2e.sh` writes the CaseList with your slide path, runs the real
CLI with `configs/region_proposal.json` and prints a ROI summary:

```bash
SLIDE=/path/to/BRACS_1003691.svs \
PYTHON=python bash components/person_c/examples/e2e/run_e2e.sh
```

Needs a GPU (NVIDIA driver >= 580), the CONCH checkpoint at
`../reference/person_c/checkpoint/conch/pytorch_model.bin` (README section 7) and the sibling
`../run/` directory. The CONCH library and the vocabulary are in the repository.

Measured result (Nano4, 1 x H200, driver 580.65.06, 8 CPUs, `pipeline_roi` env, 2026-10-27):

| Item | Value |
|---|---|
| Wall time | 5 min 4 s |
| GPU memory (allocated) | about 14.6 GB |
| CPU memory (peak RSS) | about 45.7 GB |
| Output | `E.ROIs@2.0` valid, 160 ROIs (cap 256 not reached) |
| ROI level-0 longest side | min 704 px, median 1632 px, max 14080 px; `main_info` longest side <= 2048 px |
| Asset digests | checkpoint, `third_party/CONCH` and vocabulary verified before the model loaded |

The earlier 2-ROI run on the smaller public BCSS slide `TCGA-AC-A6IW` (28777 x 34620 px, 1 min 58 s,
4.7 GB GPU) is in README section 8. Seed 42 fixes the clustering on one GPU model; other GPU models are not checked.
A checkpoint whose SHA-256
differs aborts with `AssetError`, exit code 1 and no E file (checked with a wrong file).

Failure cases covered by tests: multi-case input is rejected; a missing slide file raises
`FileNotFoundError`.
