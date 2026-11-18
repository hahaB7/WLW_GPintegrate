from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from contracts.metadata import iter_stains
from contracts.paths import resolve_run_path
from contracts.runtime import (
    cli_parser,
    load_case_list_input,
    load_config,
    validate_case_list_input,
    write_artifact,
)

from . import wsi_processing as wsi

STAIN_TYPE = "HE"          # the student was trained on H&E: every other stain keeps roi_num = 0

def _selection_event(producer: str) -> dict[str, Any]:
    return {
        "stage": "interest_pattern_extraction", "owner": "person_C",
        "artifact_contract": "E.ROIs", "action": "candidate_generated",
        "status": "selected", "selected": True,
        "reason": "interest_pattern_candidate_generated", "producer": producer,
    }


def _make_roi(case_id: str, stain_id: str, box: list[int], slide: Any,
              mpp: tuple[float, float], global_index: int, local_index: int,
              producer: str) -> dict[str, Any]:
    size = tuple(slide.dimensions)
    x, y, width, height = box
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x+width > size[0] or y+height > size[1]:
        raise ValueError(f"person_c ROI exceeds WSI bounds: {box}")
    target = wsi.target_mpp(width * height)
    main_box, main_size = wsi.main_box(box, mpp, target, size)
    identifier = wsi.roi_id(case_id, stain_id, box)
    return {
        "roi_id": identifier, "global_idx": global_index, "local_idx": local_index,
        "level0_info": wsi.roi_info(box, size, list(mpp)),
        "main_info": wsi.roi_info(main_box, main_size, target),
        "DxPair": None, "visualAttrs": None, "visualAttrs_info": None,
        "selection_history": [_selection_event(producer)],
    }


def run(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    validate_case_list_input(source, single_case=True)
    settings = wsi.validate_settings(config)
    case = deepcopy(source["case_list"][0])
    case_id = case["case_id"]
    for block in case["tissue_blocks"]:
        for stain in block["stains"]:
            stain["roi_num"] = 0
            stain["roi_list"] = []
    payload = {
        "data_mode": "inference",
        "DxItem_list": list(source["DxItem_list"]),
        "case_list": [case],
    }
    if settings["mode"] == "region_proposal":            # what produced the ROIs, for whoever reads E later
        seg = settings["segnet"]
        payload["reference_versions"] = {"interest_pattern": {
            "mode": "region_proposal",
            "segnet": {key: seg[key] for key in ("checkpoint", "roi_classes")},
            "postprocess_mpp": settings["postprocess_mpp"], "min_object_size": settings["min_object_size"],
        }}
    selected_stains = [stain for stain in iter_stains(payload) if stain["stain_type"].upper() == STAIN_TYPE]
    model = device = None
    if settings["mode"] == "region_proposal":
        from .region import figures, segnet

        for stain in selected_stains:
            if not Path(stain["filepath"]).is_file():
                raise FileNotFoundError(f"person_c WSI is missing for stain {stain['stain_id']}: {stain['filepath']}")
        if selected_stains:
            model, device = segnet.load_model(settings)
        with_image = any(settings["figures"][stage] for stage in ("tissue", "epithelium", "boxes"))
        annotations = {}
        if with_image:
            settings["figures"] = {**settings["figures"], "dir": str(resolve_run_path(settings["figures"]["dir"]))}
            if settings["figures"]["annotations"]:
                annotations = json.loads(resolve_run_path(settings["figures"]["annotations"]).read_text(encoding="utf-8"))

    global_index = 0
    for stain in iter_stains(payload):
        if stain["stain_type"].upper() != STAIN_TYPE:
            rois = []
        elif settings["mode"] == "example":
            rois = []
        else:
            import openslide

            path = Path(stain["filepath"])
            try:
                with openslide.OpenSlide(str(path)) as slide:
                    mpp = wsi.slide_mpp(slide)
                    stages = segnet.infer_stages(str(path), slide, mpp, settings, model, device, with_image=with_image)
                    boxes = wsi.drop_enclosed(wsi.level0_boxes(stages["epithelium"], mpp, slide.dimensions, settings))
                    rois = [
                        _make_roi(case_id, stain["stain_id"], box, slide, mpp,
                                  global_index + index, index, settings["component_version"])
                        for index, box in enumerate(boxes)
                    ]
                    figures.save_figures(mpp, settings["postprocess_mpp"], stages, boxes, settings["figures"], Path(stain["filename"]).stem, device,
                                        annotations.get(stain["stain_id"], ()))
            except Exception as exc:
                raise RuntimeError(f"person_c failed on stain {stain['stain_id']} ({path}): {exc}") from exc
        stain["roi_list"] = rois
        stain["roi_num"] = len(rois)
        global_index += len(rois)

    return {
        "contract": "E.ROIs", "schema_version": "2.0",
        "artifact_id": f"E-{case_id}", "case_id": case_id,
        "producer": settings["component_version"], "payload": payload,
    }


def main() -> None:
    args = cli_parser("丙: find E ROI references from one raw CaseList case").parse_args()
    if len(args.input) != 1:
        raise ValueError("interest_pattern accepts exactly one CaseList input")
    source = load_case_list_input(args.input[0], single_case=True)
    artifact = run(source, load_config(args.config))
    write_artifact(artifact, args.output, "E.ROIs")


if __name__ == "__main__":
    main()
