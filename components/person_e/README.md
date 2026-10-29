# Person E: CLEE evidence selection

Component version: `person-e/clee:1.1.0`.

This component consumes one `D.DxPairs@2.0` artifact and the matching
`H.MatchedROIs@2.0` artifact. It writes `I.CLEESelectedROIs@2.0`, preserving
all H ROIs and appending one auditable CLEE decision per diagnostic pair.

Only H events marked selected and belonging to the diagnostic pair's
`referenceWSI` are submitted. A submitted ROI receives the complete layered
`pseudo_DxPair`; `finalResult.assigned_as_ref` controls the I selected status.
An empty eligible set is a successful result and does not load the model.

## Run

From the repository or component kit root:

```bash
python -m components.person_e.clee \
  --input ../run/output/person_cde/cases/case-001/D_dx_pairs.json \
  --input ../run/output/person_cde/cases/case-001/H_matches.json \
  --output ../run/output/person_cde/cases/case-001/I_selected_rois.json \
  --config components/person_e/configs/native.json
```

The default config is a deterministic test fixture. `configs/native.json` is
the active real-inference configuration, using prefix checkpoint
`clee-261027221211` (epoch 223), its validation-calibrated threshold bundle,
and the manifest-backed `DxItem_embeddings_2607231639.pt/` catalog. MedGemma,
embedding, and CLEE checkpoint paths are managed only there and resolve below
sibling `reference/person_e/`; runtime artifacts remain below sibling `run/`.

## Image input

When `main_info.roi_path` names an existing RGB crop, CLEE uses it. Otherwise
the component opens `stains[].filepath`, reads `level0_info.xywh`, and resamples
the rectangle to `main_info.mpp`. ROI crops and relative paths must resolve below
sibling `run`; only an absolute WSI `stains[].filepath` may remain external.

## Runtime behavior

Inference uses deterministic ROI ordering and every eligible ROI. The forward
limit divides a layer into chunks; the packaged `native.json` uses 8 inputs per
forward so the MedGemma vision tower fits a 24 GB GPU without dropping ROIs.
ROIs meeting the validation-calibrated
report-concordance `case_importance.by_DxResult` threshold enter the next
layer. If one result lacks a usable result-specific threshold, the adapter
records the reason and falls back only for that result to
`case_importance.by_DxItem`; ROI-classification thresholds are never used for
evidence retention.

Selection is scoped to one case × DxItem. When its raw retained count is zero,
the terminal layer is ranked by report-conditioned evidence score descending,
then stable `global_idx`, and the configured top K is retained (native default:
1). `pseudo_DxPair.importance` remains the numeric evidence score, while
`assigned_as_ref` is the effective post-fallback decision. Raw/fallback counts,
ranks, threshold resolution, and selected ROI IDs are preserved under
`payload.reference_versions.clee_inference`; each ROI event identifies whether
selection came from the threshold or zero-candidate fallback.

Classification thresholds produce the checkpoint-declared class `assigned`
value independently from evidence selection. The current checkpoint supports
`Histologic_Type` and `Microcalcification`; their predictions are merged into
the same ROI's layered `pseudo_DxPair` without overwriting one another.

The native backend requires CUDA for the supplied configuration and fails with
a nonzero exit on missing assets, checkpoint inconsistency, model errors, or
output coverage errors. It does not import or execute DRGVLM.
