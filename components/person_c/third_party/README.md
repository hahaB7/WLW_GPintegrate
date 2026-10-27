# third_party

`CONCH/` is an unmodified copy of the runtime part of https://github.com/mahmoodlab/CONCH at commit
`141cc09c7d4ff33d8eda562bd75169b457f71a62`: `conch/open_clip_custom/` and `LICENSE`
(CC-BY-NC-ND 4.0; non-commercial use only). The checkpoint is not included (see `external-assets.yaml`).
`assets.py` checks the tree digest of `CONCH/` before the model loads, so a modified copy is rejected.
