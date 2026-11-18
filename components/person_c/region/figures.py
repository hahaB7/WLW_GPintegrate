"""One PNG per WSI and per stage, for whoever follows the segmentation (switches: ``figures`` in the config).

Stages: ``tissue`` (the multi-class tissue segmentation, one colour per class with a legend), ``epithelium`` (the processed segmentation the boxes are made from) and
``boxes`` (the level-0 boxes of the ROIs over the epithelium). Each is the slide thumbnail with the stage drawn
over it, written to ``figures.dir`` below ``run/output`` as ``<case_id>_<stain_id>_<stage>.png``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

STAGES = ("tissue", "epithelium", "boxes")
COLORS = {"epithelium": (0, 70, 255), "boxes": (255, 0, 0)}   # RGB
# colours of the classes (c9 of the project's DATA/meta/classes.json); a class that is not listed is grey
CLASS_COLORS = {"TUM": (215, 35, 55), "BEN": (240, 120, 30), "STR": (15, 155, 65), "LYM": (35, 95, 215), "NEC": (240, 180, 10),
                "FAT": (40, 190, 230), "VES": (130, 60, 190), "BLD": (110, 20, 45), "SEC": (200, 50, 160), "OTH": (25, 30, 75)}
UNLABELED = 255
WIDTH = 1600


def overlay(thumb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.55) -> np.ndarray:
    """Fill ``mask`` (any shape, scaled to the thumbnail) with ``color`` and draw its outline."""

    import cv2

    full = cv2.resize(mask.astype(np.uint8), (thumb.shape[1], thumb.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    out = thumb.copy()
    out[full] = ((1 - alpha) * thumb[full] + alpha * np.array(color)).astype(np.uint8)
    contours, _ = cv2.findContours(full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 0), 1)
    return out


def class_overlay(thumb: np.ndarray, labels: np.ndarray, classes: Sequence[str], alpha: float = 0.6) -> np.ndarray:
    """Paint every pixel of ``labels`` (index into ``classes``, UNLABELED left as the slide) with the colour of its class
    and add a legend of the classes that occur."""

    import cv2

    full = cv2.resize(labels, (thumb.shape[1], thumb.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = thumb.copy()
    present = []
    for index, name in enumerate(classes):
        sel = full == index
        if not sel.any():
            continue
        present.append(name)
        color = np.array(CLASS_COLORS.get(name, (128, 128, 128)))
        out[sel] = ((1 - alpha) * thumb[sel] + alpha * color).astype(np.uint8)
    for row, name in enumerate(present):
        y = 60 + row * 34
        cv2.rectangle(out, (12, y), (40, y + 24), CLASS_COLORS.get(name, (128, 128, 128)), -1)
        cv2.rectangle(out, (12, y), (40, y + 24), (0, 0, 0), 1)
        cv2.putText(out, name, (50, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return out


def draw_boxes(image: np.ndarray, boxes: Sequence[Sequence[int]], scale: float) -> np.ndarray:
    """Draw level-0 ``[x, y, w, h]`` boxes on an image that is ``scale`` times the level-0 size."""

    import cv2

    out = image.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(out, (int(x * scale), int(y * scale)), (int((x + w) * scale), int((y + h) * scale)), COLORS["boxes"], 2)
    return out


def caption(image: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = image.copy()
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
    cv2.putText(out, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def save_figures(slide: Any, stages: Mapping[str, Any], boxes: Sequence[Sequence[int]], figures: Mapping[str, Any],
                 case_id: str, stain_id: str) -> list[str]:
    """Write the PNGs whose switch is on; return their paths. A file appears whole or not at all."""

    import cv2

    wanted = [stage for stage in STAGES if figures.get(stage)]
    if not wanted:
        return []
    thumb = np.asarray(slide.get_thumbnail((WIDTH, WIDTH)).convert("RGB"))
    scale = thumb.shape[1] / slide.dimensions[0]
    directory = Path(figures["dir"])
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for stage in wanted:
        if stage == "boxes":
            image = draw_boxes(overlay(thumb, stages["epithelium"], COLORS["epithelium"], 0.3), boxes, scale)
        elif stage == "tissue":
            image = class_overlay(thumb, stages["tissue"], stages["classes"])
        else:
            image = overlay(thumb, stages[stage], COLORS[stage])
        image = caption(image, f"{stain_id} {stage}" + (f" ({len(boxes)})" if stage == "boxes" else ""))
        path = directory / f"{case_id}_{stain_id}_{stage}.png"
        temp = path.with_name(path.stem + ".tmp.png")
        try:
            if not cv2.imwrite(str(temp), image[:, :, ::-1]):
                raise OSError(f"could not write {temp}")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        paths.append(str(path))
    return paths
