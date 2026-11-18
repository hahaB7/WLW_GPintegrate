"""Person C: WSI interest pattern extraction. One CaseList case in, ``E.ROIs@2.0`` out.

For every H&E stain of the case the SEGNET student (``segmentation.py``) labels the slide, the epithelium (TUM and BEN, holes
filled) is cut into connected objects, and the minimum bounding box of each object is one ROI. The file has five blocks:

1. Settings    the config of ``configs/*.json``, every key checked here
2. Boxes       epithelium mask -> level-0 boxes
3. ROI         the ``level0_info`` / ``main_info`` records of a box
4. Case        CaseList -> E, one stain after the other
5. CLI         ``python -m components.person_c.interest_pattern --input ... --output ... --config ...``
"""

from __future__ import annotations

import json
from copy import deepcopy
from hashlib import sha256
from math import ceil, floor, isfinite
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

STAIN_TYPE = "HE"          # the student was trained on H&E: every other stain keeps roi_num = 0


# ============================================================ 1. Settings
def positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a positive number")
    result = float(value)
    if not isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_settings(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("person_c config must be a JSON object")
    version = config.get("component_version")
    if not isinstance(version, str) or not version:
        raise ValueError("person_c config requires component_version")
    mode = config.get("mode")
    if mode not in {"example", "region_proposal"}:
        raise ValueError("person_c mode must be example or region_proposal")
    result = {**config, "mode": mode}
    if mode == "example":                                  # no slide is opened and no ROI is made
        return result
    result["segnet"] = validate_segnet(result.get("segnet"))
    result["figures"] = validate_figures(result.get("figures"))
    result["min_object_size"] = integer(result.get("min_object_size"), "min_object_size", 1)
    result["postprocess_mpp"] = positive_number(result.get("postprocess_mpp"), "postprocess_mpp")
    return result


def validate_segnet(cfg: Any) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ValueError("person_c segnet requires a segnet object")
    checkpoint = cfg.get("checkpoint")
    if (not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("path"), str) or not checkpoint["path"]
            or not isinstance(checkpoint.get("revision"), str) or not checkpoint["revision"].startswith("sha256:")):
        raise ValueError("person_c segnet.checkpoint must be {path, revision: sha256:<digest>}")
    classes = cfg.get("roi_classes")
    if not isinstance(classes, list) or not classes or any(not isinstance(c, str) or not c for c in classes):
        raise ValueError("person_c segnet.roi_classes must be a non-empty string list")
    return {"checkpoint": checkpoint, "roi_classes": classes}


def _bounded_int(value: Any, low: int, high: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"person_c {name} must be an integer in [{low}, {high}]")
    return value


def validate_figures(cfg: Any) -> dict[str, Any]:
    """``figures``: one switch per stage (tissue, epithelium, boxes) and the directory below sibling ``run/output``."""

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("person_c figures must be an object")
    result = {"dir": cfg.get("dir"), "annotations": cfg.get("annotations")}
    if result["annotations"] is not None and not (isinstance(result["annotations"], str) and result["annotations"]):
        raise ValueError("person_c figures.annotations must be a path or absent")
    for stage in ("tissue", "epithelium", "boxes"):
        if not isinstance(cfg.get(stage, False), bool):
            raise ValueError(f"person_c figures.{stage} must be boolean")
        result[stage] = cfg.get(stage, False)
    result["downscale"] = _bounded_int(cfg.get("downscale", 1), 1, 64, "figures.downscale")
    if any(result[stage] for stage in ("tissue", "epithelium", "boxes")) and not result["dir"]:
        raise ValueError("person_c figures.dir is required when a figure is switched on")
    return result


# ============================================================ 2. Boxes
def slide_mpp(slide: Any) -> tuple[float, float]:
    import openslide

    x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
    try:
        return positive_number(float(x), "WSI mpp_x"), positive_number(float(y), "WSI mpp_y")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"WSI MPP is missing or invalid: x={x!r}, y={y!r}") from exc


def level0_boxes(mask: Any, mpp: tuple[float, float], size: tuple[int, int], settings: dict[str, Any]) -> list[list[int]]:
    """Level-0 ``[x, y, w, h]`` boxes of the connected objects (4-connectivity) of ``mask`` (boolean, at
    ``postprocess_mpp``): objects of fewer than ``min_object_size`` pixels get no box, and each box is the minimum
    bounding box, floored / ceiled to level 0 and clamped to the slide."""

    import cv2
    import numpy as np

    target = settings["postprocess_mpp"]
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    width, height = size
    boxes = []
    for left, top, w, h, area in stats[1:]:                  # row 0 is the background
        if area < settings["min_object_size"]:
            continue
        x0, y0 = max(0, min(width, floor(left*target/mpp[0]))), max(0, min(height, floor(top*target/mpp[1])))
        x1, y1 = max(0, min(width, ceil((left+w)*target/mpp[0]))), max(0, min(height, ceil((top+h)*target/mpp[1])))
        if x1 > x0 and y1 > y0:
            boxes.append([x0, y0, x1-x0, y1-y0])
    return boxes


def drop_enclosed(boxes: list[list[int]]) -> list[list[int]]:
    """Drop every box that lies completely inside another box (the same box twice counts once)."""
    ordered = sorted(boxes, key=lambda b: (-b[2] * b[3], b[1], b[0]))
    kept: list[list[int]] = []
    for x, y, w, h in ordered:
        if not any(kx <= x and ky <= y and x + w <= kx + kw and y + h <= ky + kh for kx, ky, kw, kh in kept):
            kept.append([x, y, w, h])
    return sorted(kept, key=lambda b: (b[1], b[0]))


# ============================================================ 3. ROI
def target_mpp(area: int) -> float:
    for cutoff, mpp in ((1, 0.25), (4, 0.5), (16, 1.0), (64, 2.0), (256, 4.0), (1024, 8.0)):
        if area <= cutoff * 1_000_000:
            return mpp
    return 16.0


def normalized_box(box: list[int], size: tuple[int, int]) -> list[float]:
    x, y, width, height = box
    full_width, full_height = size
    return [(x+width/2)/full_width, (y+height/2)/full_height,
            width/full_width, height/full_height]


def main_box(box: list[int], mpp: tuple[float, float], target: float,
             size: tuple[int, int]) -> tuple[list[int], tuple[int, int]]:
    full_width, full_height = size
    main_width, main_height = ceil(full_width*mpp[0]/target), ceil(full_height*mpp[1]/target)
    x, y, width, height = box
    x0, y0 = floor(x*mpp[0]/target), floor(y*mpp[1]/target)
    x1, y1 = min(main_width, ceil((x+width)*mpp[0]/target)), min(main_height, ceil((y+height)*mpp[1]/target))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"person_c ROI collapsed at main MPP: {box}")
    return [x0, y0, x1-x0, y1-y0], (main_width, main_height)


def roi_info(box: list[int], size: tuple[int, int], mpp: Any) -> dict[str, Any]:
    return {"area": box[2]*box[3], "coords_seg": None,
            "cxcywh": normalized_box(box, size), "mpp": mpp,
            "roi_path": None, "roi_wh": box[2:], "xywh": box}


def roi_id(case_id: str, stain_id: str, box: list[int]) -> str:
    key = f"{case_id}\0{stain_id}\0{','.join(map(str, box))}"
    return f"{case_id}-{stain_id}-roi-{sha256(key.encode()).hexdigest()[:16]}"


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
    target = target_mpp(width * height)
    working_box, working_size = main_box(box, mpp, target, size)
    identifier = roi_id(case_id, stain_id, box)
    return {
        "roi_id": identifier, "global_idx": global_index, "local_idx": local_index,
        "level0_info": roi_info(box, size, list(mpp)),
        "main_info": roi_info(working_box, working_size, target),
        "DxPair": None, "visualAttrs": None, "visualAttrs_info": None,
        "selection_history": [_selection_event(producer)],
    }


# ============================================================ 4. Case
def run(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    validate_case_list_input(source, single_case=True)
    settings = validate_settings(config)
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
        from . import segmentation

        for stain in selected_stains:
            if not Path(stain["filepath"]).is_file():
                raise FileNotFoundError(f"person_c WSI is missing for stain {stain['stain_id']}: {stain['filepath']}")
        if selected_stains:
            model, device = segmentation.load_model(settings)
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
                    mpp = slide_mpp(slide)
                    stages = segmentation.infer_stages(str(path), slide, mpp, settings, model, device, with_image=with_image)
                    boxes = drop_enclosed(level0_boxes(stages["epithelium"], mpp, slide.dimensions, settings))
                    rois = [
                        _make_roi(case_id, stain["stain_id"], box, slide, mpp,
                                  global_index + index, index, settings["component_version"])
                        for index, box in enumerate(boxes)
                    ]
                    segmentation.save_figures(mpp, settings["postprocess_mpp"], stages, boxes, settings["figures"], Path(stain["filename"]).stem, device,
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


# ============================================================ 5. CLI
def main() -> None:
    args = cli_parser("丙: find E ROI references from one raw CaseList case").parse_args()
    if len(args.input) != 1:
        raise ValueError("interest_pattern accepts exactly one CaseList input")
    source = load_case_list_input(args.input[0], single_case=True)
    artifact = run(source, load_config(args.config))
    write_artifact(artifact, args.output, "E.ROIs")


if __name__ == "__main__":
    main()
