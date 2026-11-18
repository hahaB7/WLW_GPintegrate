"""Settings, slide geometry and ROI records of person C's interest extractor (the mask itself comes from region/segnet.py)."""

from __future__ import annotations

from hashlib import sha256
from math import ceil, floor, isfinite
from pathlib import Path
from typing import Any


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
    whitelist = result.get("stain_type_whitelist", ["HE"])
    if (
        not isinstance(whitelist, list)
        or not whitelist
        or any(not isinstance(value, str) or not value.strip() for value in whitelist)
    ):
        raise ValueError("person_c stain_type_whitelist must be a non-empty string list")
    normalized_whitelist = [value.strip().upper() for value in whitelist]
    if len(normalized_whitelist) != len(set(normalized_whitelist)):
        raise ValueError("person_c stain_type_whitelist contains duplicates")
    result["stain_type_whitelist"] = normalized_whitelist
    if mode == "example":                                  # no slide is opened: the fixed boxes of the config, or none
        result.setdefault("demo_regions", [])
        result.setdefault("demo_mpp", {"x": 0.5, "y": 0.5})
        if not isinstance(result["demo_regions"], list) or not isinstance(result["demo_mpp"], dict):
            raise ValueError("person_c example demo_regions must be a list and demo_mpp an object")
        return result
    result["segnet"] = validate_segnet(result.get("segnet"))
    result["figures"] = validate_figures(result.get("figures"))
    result["min_object_size"] = integer(result.get("min_object_size"), "min_object_size", 1)
    result["postprocess_mpp"] = positive_number(result.get("postprocess_mpp"), "postprocess_mpp")
    if result.get("device", "auto") not in {"auto", "cpu", "cuda"}:
        raise ValueError("person_c device must be auto, cpu, or cuda")
    result["device"] = result.get("device", "auto")
    if not isinstance(result.get("save_roi_png", False), bool):
        raise ValueError("person_c save_roi_png must be boolean")
    result["save_roi_png"] = result.get("save_roi_png", False)
    if result["save_roi_png"] and not result.get("roi_output_dir"):
        raise ValueError("person_c roi_output_dir is required when save_roi_png is true")
    return result


def validate_segnet(cfg: Any) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ValueError("person_c segnet requires a segnet object")
    checkpoints = cfg.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints or any(
        not isinstance(c, dict) or not isinstance(c.get("path"), str) or not c["path"]
        or not isinstance(c.get("revision"), str) or not c["revision"].startswith("sha256:")
        for c in checkpoints
    ):
        raise ValueError("person_c segnet.checkpoints must be a non-empty list of {path, revision: sha256:<digest>}")
    classes = cfg.get("roi_classes")
    if not isinstance(classes, list) or not classes or any(not isinstance(c, str) or not c for c in classes):
        raise ValueError("person_c segnet.roi_classes must be a non-empty string list")
    result = {**cfg}
    for name in ("tile_um", "margin_um"):
        result[name] = integer(cfg.get(name), f"segnet.{name}", 4)
        if result[name] % 4:
            raise ValueError(f"person_c segnet.{name} must be a multiple of 4")
    result["tta_views"] = integer(cfg.get("tta_views"), "segnet.tta_views", 1)
    if result["tta_views"] > 8:
        raise ValueError("person_c segnet.tta_views must be 1..8")
    result["batch_size"] = integer(cfg.get("batch_size"), "segnet.batch_size", 1)
    result["num_workers"] = integer(cfg.get("num_workers"), "segnet.num_workers", 0)
    return result


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
    result["quality"] = _bounded_int(cfg.get("quality", 90), 1, 100, "figures.quality")
    thickness = cfg.get("box_thickness")
    result["box_thickness"] = 2 if thickness is None else _bounded_int(thickness, 1, 20, "figures.box_thickness")
    if any(result[stage] for stage in ("tissue", "epithelium", "boxes")) and not result["dir"]:
        raise ValueError("person_c figures.dir is required when a figure is switched on")
    return result


def slide_mpp(slide: Any) -> tuple[float, float]:
    import openslide

    x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
    try:
        return positive_number(float(x), "WSI mpp_x"), positive_number(float(y), "WSI mpp_y")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"WSI MPP is missing or invalid: x={x!r}, y={y!r}") from exc


def read_level(slide: Any, mpp: tuple[float, float], target_mpp: float) -> int:
    candidates = [
        level for level, downsample in enumerate(slide.level_downsamples)
        if all(axis * float(downsample) <= target_mpp * 1.01 for axis in mpp)
    ]
    return max(candidates) if candidates else 0


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


def roi_info(box: list[int], size: tuple[int, int], mpp: Any,
             path: str | None = None) -> dict[str, Any]:
    return {"area": box[2]*box[3], "coords_seg": None,
            "cxcywh": normalized_box(box, size), "mpp": mpp,
            "roi_path": path, "roi_wh": box[2:], "xywh": box}


def roi_id(case_id: str, stain_id: str, box: list[int]) -> str:
    key = f"{case_id}\0{stain_id}\0{','.join(map(str, box))}"
    return f"{case_id}-{stain_id}-roi-{sha256(key.encode()).hexdigest()[:16]}"


def save_png(slide: Any, box: list[int], output_size: tuple[int, int], target: float,
             mpp: tuple[float, float], output_dir: Path, identifier: str) -> str:
    from PIL import Image

    x, y, width, height = box
    level = read_level(slide, mpp, target)
    downsample = float(slide.level_downsamples[level])
    region = slide.read_region((x, y), level, (max(1, ceil(width/downsample)), max(1, ceil(height/downsample))))
    try:
        rgb = region.convert("RGB")
        try:
            if rgb.size != output_size:
                resized = rgb.resize(output_size, Image.Resampling.BICUBIC)
                rgb.close()
                rgb = resized
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / (sha256(identifier.encode()).hexdigest()[:24] + ".png")
            temp_path = path.with_suffix(".png.tmp")
            try:
                rgb.save(temp_path, format="PNG")
                temp_path.replace(path)
            finally:
                temp_path.unlink(missing_ok=True)
            return str(path)
        finally:
            rgb.close()
    finally:
        region.close()
