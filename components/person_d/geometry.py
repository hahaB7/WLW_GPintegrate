"""E ROI geometry rules shared by example and native mode.

Standard library only, so the contract-only example mode can validate geometry
without image or model dependencies.  Bounds against the WSI itself can only be
checked when the slide is opened (``roi_reader.read_wsi_roi``).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _mpp_pair(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        pair = (float(value), float(value))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        pair = (float(value[0]), float(value[1]))
    else:
        raise ValueError(f"Invalid {name}: {value!r}")
    if any(not math.isfinite(part) or part <= 0 for part in pair):
        raise ValueError(f"{name} must contain positive finite values: {value!r}")
    return pair


def roi_geometry(roi: Mapping[str, Any]) -> tuple[tuple[int, int, int, int], tuple[int, int], float]:
    """Validate E geometry and return level-0 xywh, target size, and target downsample."""

    roi_id = roi["roi_id"]
    x, y, width, height = (int(value) for value in roi["level0_info"]["xywh"])
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"ROI {roi_id} has invalid level0_info.xywh")
    target_w, target_h = (int(value) for value in roi["main_info"]["roi_wh"])
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"ROI {roi_id} has invalid main_info.roi_wh")

    level0_mpp = _mpp_pair(roi["level0_info"]["mpp"], "level0_info.mpp")
    target_mpp = _mpp_pair(roi["main_info"]["mpp"], "main_info.mpp")
    expected = (
        round(width * level0_mpp[0] / target_mpp[0]),
        round(height * level0_mpp[1] / target_mpp[1]),
    )
    if abs(expected[0] - target_w) > 1 or abs(expected[1] - target_h) > 1:
        raise ValueError(
            f"ROI {roi_id} main_info.roi_wh {target_w, target_h} does not match "
            f"level0 xywh and mpp (expected about {expected})"
        )
    downsample = min(target_mpp[0] / level0_mpp[0], target_mpp[1] / level0_mpp[1])
    return (x, y, width, height), (target_w, target_h), downsample
