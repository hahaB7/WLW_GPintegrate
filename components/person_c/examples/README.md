# Person C example command

為遵守 runtime I/O 只放 sibling `run` 的規則，本目錄不再保存 input/output JSON 或第二份 config。
`command.txt` 直接讀取單案 CaseList fixture，並將結果寫入 `run/output/work`。真實 inference
使用 `configs/region_proposal.json`。

`e2e/run_e2e.sh` runs the real mode `region_proposal` on one slide (`SLIDE=/path/to/slide.svs bash components/person_c/examples/e2e/run_e2e.sh`,
a GPU and `reference/person_c/checkpoint/segnet/c9_sclip_bracs_s0.pt` are needed) and prints the number of ROIs and the
longest side of the smallest and the largest. To see the stages, switch the PNGs on in a copy of `configs/region_proposal.json` (`figures`).

