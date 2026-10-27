# Person C configuration

`example.json` only preserves the shared `--config` shape (stub mode).

`region_proposal.json`:

| Key | Meaning |
|---|---|
| `stain_types` | stain types to process; `null` = all. Others keep `roi_list = []` |
| `region.conch_checkpoint`, `vocab_json` | external assets (checkpoint, BCSS class descriptions) below `reference/person_c/`; sha256 checked before the model loads |
| `region.conch_lib_path` | CONCH library shipped inside the component (path relative to `components/person_c/`); sha256 checked |
| `region.vocab`, `proposal_categories` | zero-shot vocab key (`BCSS`) and kept categories (`[]` = vocab default: tumor, dcis, normal_acinus_or_duct, metaplasia_NOS) |
| `region.fields`, `mag`, `overlap` | CONCH field(s) of view in level-0 px, magnification, crop overlap |
| `region.K`, `radius`, `pca_dim`, `use_patch`, `seed`, `zs_temp` | vMF clustering / zero-shot parameters |
| `region.device`, `batch_size`, `num_workers` | runtime |
| `roi.target_mpp` | MPP of `main_info` (default working resolution) |
| `roi.max_main_side_px` | longest side of `main_info`; larger ROIs use a coarser MPP (`level0_info` unchanged) |
| `roi.min_region_area_px` | drop polygons smaller than this (level-0 px^2) |
| `roi.max_rois_per_stain` | cap per stain (largest polygons first) |
