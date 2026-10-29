from __future__ import annotations

from collections.abc import Callable
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

def _selection_event(producer: str) -> dict[str, Any]:
    return {
        "stage": "interest_pattern_extraction", "owner": "person_C",
        "artifact_contract": "E.ROIs", "action": "candidate_generated",
        "status": "selected", "selected": True,
        "reason": "interest_pattern_candidate_generated", "producer": producer,
    }


def _make_roi(case_id: str, stain_id: str, box: list[int], slide: Any,
              mpp: tuple[float, float], global_index: int, local_index: int,
              settings: dict[str, Any]) -> dict[str, Any]:
    size = tuple(slide.dimensions)
    x, y, width, height = box
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x+width > size[0] or y+height > size[1]:
        raise ValueError(f"person_c ROI exceeds WSI bounds: {box}")
    target = wsi.target_mpp(width * height)
    main_box, main_size = wsi.main_box(box, mpp, target, size)
    identifier = wsi.roi_id(case_id, stain_id, box)
    path = None
    if settings["save_roi_png"]:
        output_dir = resolve_run_path(settings["roi_output_dir"])
        path = wsi.save_png(slide, box, tuple(main_box[2:]), target, mpp,
                            output_dir, identifier)
    return {
        "roi_id": identifier, "global_idx": global_index, "local_idx": local_index,
        "level0_info": wsi.roi_info(box, size, list(mpp)),
        "main_info": wsi.roi_info(main_box, main_size, target, path),
        "DxPair": None, "visualAttrs": None, "visualAttrs_info": None,
        "selection_history": [_selection_event(settings["component_version"])],
    }


def _demo_rois(case_id: str, stain: dict[str, Any], settings: dict[str, Any],
               first_index: int) -> list[dict[str, Any]]:
    """Retain the historical pipeline fixture, which has no mounted WSI."""
    mpp_config = settings["demo_mpp"]
    mpp = (wsi.positive_number(mpp_config.get("x"), "demo_mpp.x"),
           wsi.positive_number(mpp_config.get("y"), "demo_mpp.y"))
    boxes = []
    for region in settings["demo_regions"]:
        if not isinstance(region, dict):
            raise ValueError("person_c demo_regions entries must be objects")
        boxes.append([wsi.integer(region.get("x"), "demo x"),
                      wsi.integer(region.get("y"), "demo y"),
                      wsi.integer(region.get("width"), "demo width", 1),
                      wsi.integer(region.get("height"), "demo height", 1)])
    size = (max((box[0]+box[2] for box in boxes), default=1),
            max((box[1]+box[3] for box in boxes), default=1))
    return [{
        "roi_id": wsi.roi_id(case_id, stain["stain_id"], box),
        "global_idx": first_index + index, "local_idx": index,
        "level0_info": wsi.roi_info(box, size, list(mpp)),
        "main_info": wsi.roi_info(box, size, mpp[0]),
        "DxPair": None, "visualAttrs": None, "visualAttrs_info": None,
        "selection_history": [_selection_event(settings["component_version"])],
    } for index, box in enumerate(boxes)]


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
    allowed_stains = set(settings["stain_type_whitelist"])
    model = device = None
    if settings["backend"] == "mrcps":
        selected_stains = [
            stain
            for stain in iter_stains(payload)
            if stain["stain_type"].upper() in allowed_stains
        ]
        for stain in selected_stains:
            path = Path(stain["filepath"])
            if not path.is_file():
                raise FileNotFoundError(f"person_c WSI is missing for stain {stain['stain_id']}: {path}")
        if selected_stains:
            model, device = wsi.load_model(settings)

    global_index = 0
    for stain in iter_stains(payload):
        if stain["stain_type"].upper() not in allowed_stains:
            rois = []
        elif settings["backend"] == "demo":
            rois = _demo_rois(case_id, stain, settings, global_index)
        else:
            import openslide

            path = Path(stain["filepath"])
            try:
                with openslide.OpenSlide(str(path)) as slide:
                    mpp = wsi.slide_mpp(slide)
                    mask = wsi.infer_mask(slide, mpp, settings, model, device)
                    boxes = wsi.level0_boxes(mask, mpp, slide.dimensions, settings)
                    rois = [
                        _make_roi(case_id, stain["stain_id"], box, slide, mpp,
                                  global_index + index, index, settings)
                        for index, box in enumerate(boxes)
                    ]
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


def build_example_artifact(source: dict[str, Any], producer: str) -> dict[str, Any]:
    case = deepcopy(source["case_list"][0])
    for block in case["tissue_blocks"]:
        for stain in block["stains"]:
            stain["roi_num"] = 0
            stain["roi_list"] = []
    return _artifact(
        source,
        case,
        producer,
        f"E-example-{case['case_id']}",
        {"example_stub": {"implemented": False, "purpose": "contract smoke test"}},
    )


def _region_cfg(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve assets and check their pinned digests.

    ``conch_checkpoint`` and ``vocab_json`` (the class descriptions) are external, below
    reference/person_c. The CONCH library ships inside the component (path relative to this package).
    """

    region = dict(config["region"])
    for key in ("conch_checkpoint", "vocab_json"):
        path = resolve_person_reference_path("person_c", region[key])
        if not path.exists():
            raise FileNotFoundError(f"person_c reference asset is missing: {path}")
        region[key] = str(path)
    package = Path(__file__).resolve().parent
    lib = (package / region["conch_lib_path"]).resolve()
    if not lib.is_relative_to(package):
        raise ValueError(f"region.conch_lib_path must stay inside the component: {lib}")
    region["conch_lib_path"] = str(lib)
    verify_assets(region)
    return region


def slide_info(path: str) -> tuple[tuple[int, int], tuple[float, float]]:
    """Return ``((width, height), (mpp_x, mpp_y))`` of level 0; fail if MPP is absent."""

    import openslide

    if not Path(path).is_file():
        raise FileNotFoundError(f"WSI is missing: {path}")
    with openslide.OpenSlide(path) as slide:
        mpp_x = slide.properties.get("openslide.mpp-x")
        mpp_y = slide.properties.get("openslide.mpp-y")
        if not mpp_x or not mpp_y:
            raise ValueError(f"WSI has no openslide.mpp-x/mpp-y metadata: {path}")
        width, height = slide.level_dimensions[0]
    mpp = (float(mpp_x), float(mpp_y))
    if any(value <= 0 for value in mpp):
        raise ValueError(f"WSI has a non-positive MPP {mpp}: {path}")
    return (int(width), int(height)), mpp


def build_region_artifact(
    source: dict[str, Any],
    config: Mapping[str, Any],
    *,
    propose: Propose,
    slide_info: SlideInfo,
) -> dict[str, Any]:
    producer = _producer(config)
    region = dict(config["region"])
    roi_cfg = config["roi"]
    stain_types = config.get("stain_types")
    case = deepcopy(source["case_list"][0])
    global_idx = 0
    for block in case["tissue_blocks"]:
        for stain in block["stains"]:
            stain["roi_list"] = []
            if stain_types is not None and stain["stain_type"] not in stain_types:
                stain["roi_num"] = 0
                continue
            slide_wh, mpp = slide_info(stain["filepath"])
            boxes = plan_boxes(
                propose(stain["filepath"], region),
                slide_wh,
                float(roi_cfg["min_region_area_px"]),
                int(roi_cfg["max_rois_per_stain"]),
            )
            for local_idx, box in enumerate(boxes):
                level0, main = build_roi_info(
                    box,
                    mpp,
                    float(roi_cfg["target_mpp"]),
                    int(roi_cfg["max_main_side_px"]),
                )
                stain["roi_list"].append(
                    {
                        "roi_id": f"{stain['stain_id']}-roi-{local_idx:05d}",
                        "global_idx": global_idx,
                        "local_idx": local_idx,
                        "level0_info": level0,
                        "main_info": main,
                        "DxPair": None,
                        "visualAttrs": None,
                        "visualAttrs_info": None,
                        "selection_history": [candidate_event(producer)],
                    }
                )
                global_idx += 1
            stain["roi_num"] = len(stain["roi_list"])
    return _artifact(
        source,
        case,
        producer,
        f"E-{case['case_id']}",
        {"interest_pattern": {"region": region, "roi": dict(roi_cfg)}},
    )


def main() -> None:
    args = cli_parser("丙: find E ROI references from one raw CaseList case").parse_args()
    if len(args.input) != 1:
        raise ValueError("interest_pattern accepts exactly one CaseList input")
    source = load_case_list_input(args.input[0], single_case=True)
    artifact = run(source, load_config(args.config))
    write_artifact(artifact, args.output, "E.ROIs")


if __name__ == "__main__":
    main()
