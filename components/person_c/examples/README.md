# Person C examples

| File | What it is |
|---|---|
| `case_list.valid.json` | A single-case `CaseListInput@1.0` with one H&E and one IHC stain (the IHC stain is not processed). |
| `E_rois.expected.json` | The `E.ROIs@2.0` that mode `region_proposal` writes for that input when the slide and the epithelium mask are the made-up ones of `canonical_output()` in `tests/test_interest_pattern.py`: a ring with an object in its hole (dropped, enclosed), an object at the slide border, an object that is too small (no box) give two ROIs. |
| `e2e/run_e2e.sh` | The real mode on one slide: `SLIDE=/path/to/slide.svs bash components/person_c/examples/e2e/run_e2e.sh` (a GPU and the checkpoint under `reference/person_c/checkpoint/segnet/` are needed); it prints the number of ROIs and the longest side of the smallest and the largest. |

The expected E is compared with the code's output by `CanonicalExampleTests` (`python -m unittest discover -s components/person_c/tests -t .`);
a change of the ROI rules shows up there. To see the stages of a real slide, copy `configs/region_proposal.json`, switch the
`figures` on and run it as in `README.md`.

No input or output JSON of a real run is kept here (the runtime rules put them below the sibling `run/`), and no slide or
checkpoint is committed (`external-assets.yaml` lists them).
