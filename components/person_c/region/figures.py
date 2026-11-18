"""One JPEG per WSI and per stage, for whoever follows the segmentation (switches: ``figures`` in the config).

Stages: ``tissue`` (the multi-class tissue segmentation, one colour per class with a legend), ``epithelium`` (the processed
segmentation the boxes are made from) and ``boxes`` (the level-0 boxes of the ROIs over the epithelium, dashed lines). Each is the slide at
the resolution of the segmentation (``postprocess_mpp``, 1 um/px) with the stage drawn over it, written to ``figures.dir``
below ``run/output`` as ``<name>_<stage>.jpg`` (``name``: the stem of the slide's file name, ``BRACS_1228``). The slide image is the one the student already read; the
painting runs on the GPU in bands of rows, so that the memory stays at a few images of the final size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

STAGES = ("tissue", "epithelium", "boxes")
EPITHELIUM = (0, 70, 255)                    # RGB
ANNOTATION = (0, 170, 0)                     # BGR, green: the ROIs of an annotation
BOX = (0, 0, 255)                            # BGR: the images of this module are BGR from ``paint`` on
# colours of the classes (c9 of the project's DATA/meta/classes.json); a class that is not listed is grey
CLASS_COLORS = {"TUM": (215, 35, 55), "BEN": (240, 120, 30), "STR": (15, 155, 65), "LYM": (35, 95, 215), "NEC": (240, 180, 10),
                "FAT": (40, 190, 230), "VES": (130, 60, 190), "BLD": (110, 20, 45), "SEC": (200, 50, 160), "OTH": (25, 30, 75)}
UNLABELED = 255
BOX_THICKNESS = 4                            # px of the figure: a thin line, found by its colour and its dark edge
BAND = 1024                                  # rows per band
QUALITY = 90                                 # JPEG quality: clear to the eye, a tenth of the size of a PNG
ALPHA = {"tissue": 0.6, "epithelium": 0.55, "boxes": 0.3}


def paint(stage: str, band: np.ndarray, labels: np.ndarray, epithelium: np.ndarray, classes: Sequence[str], device: Any) -> np.ndarray:
    """The stage drawn on a band of the slide (uint8 RGB, rows of the stage arrays) -> the painted band, as BGR. On the GPU."""

    import torch
    import torch.nn.functional as F

    image = torch.from_numpy(band).to(device)
    epi = torch.from_numpy(epithelium).to(device)
    if stage == "tissue":
        lab = torch.from_numpy(labels).to(device)
        lut = torch.zeros((256, 3), dtype=torch.float32, device=device)
        for index, name in enumerate(classes):
            lut[index] = torch.tensor(CLASS_COLORS.get(name, (128, 128, 128)), dtype=torch.float32)
        color, mask = lut[lab.long()], lab != UNLABELED
    else:
        color, mask = torch.tensor(EPITHELIUM, dtype=torch.float32, device=device), epi
    alpha = ALPHA[stage]
    mixed = ((1 - alpha) * image.float() + alpha * color).to(torch.uint8)
    image = torch.where(mask[..., None], mixed, image)
    if stage == "epithelium":                                   # a black line of about 3 px along the border
        inner = -F.max_pool2d(-epi[None, None].float(), 7, 1, 3)[0, 0] > 0.5
        image = torch.where((epi & ~inner)[..., None], torch.zeros_like(image), image)
    return image.flip(-1).cpu().numpy()                        # BGR, the order OpenCV draws and writes in


def legend(image: np.ndarray, classes: Sequence[str], present: Sequence[bool], unit: float) -> None:
    import cv2

    names = [name for name, here in zip(classes, present) if here]
    for row, name in enumerate(names):
        y, color = int((60 + row * 34) * unit), tuple(CLASS_COLORS.get(name, (128, 128, 128)))[::-1]
        cv2.rectangle(image, (int(12 * unit), y), (int(40 * unit), y + int(24 * unit)), color, -1)
        cv2.rectangle(image, (int(12 * unit), y), (int(40 * unit), y + int(24 * unit)), (0, 0, 0), max(1, int(unit)))
        cv2.putText(image, name, (int(50 * unit), y + int(20 * unit)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * unit, (0, 0, 0), max(1, int(2 * unit)))


def caption(image: np.ndarray, text: str, unit: float) -> None:
    import cv2

    for color, width in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(image, text, (int(12 * unit), int(32 * unit)), cv2.FONT_HERSHEY_SIMPLEX, unit, color, max(1, int(width * unit)))


def dashed_line(image: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], thickness: int) -> None:
    """In place: a dashed line (dash 6 and gap 6 line widths, so that the slide stays visible between the dashes), with a dark
    edge one pixel wide on each side, so that it stands out on a light and on a dark background alike."""

    import cv2

    length = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    period, dash = 12 * thickness, 6 * thickness
    for t in range(0, length, period):
        a, b = t / length, min(t + dash, length) / length
        p, q = ((round(start[0] + (end[0] - start[0]) * t), round(start[1] + (end[1] - start[1]) * t)) for t in (a, b))
        cv2.line(image, p, q, (0, 0, 0), thickness + 2)
        cv2.line(image, p, q, color, thickness)


def draw_boxes(image: np.ndarray, boxes: Sequence[Sequence[int]], mpp0: tuple[float, float], mpp: float, thickness: int,
               color: tuple[int, int, int] = BOX) -> None:
    """In place: level-0 ``[x, y, w, h]`` boxes on an image of ``mpp`` um/px, as dashed rectangles."""

    for x, y, w, h in boxes:
        x0, y0 = round(x * mpp0[0] / mpp), round(y * mpp0[1] / mpp)
        x1, y1 = round((x + w) * mpp0[0] / mpp), round((y + h) * mpp0[1] / mpp)
        for start, end in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            dashed_line(image, start, end, color, thickness)


def mark_annotations(image: np.ndarray, boxes: Sequence[Sequence[int]], mpp0: tuple[float, float], mpp: float, unit: float) -> None:
    """In place: a ring and a number around every annotation ROI, so that it is found at a glance in a zoomed-out figure.

    The ring lies outside the box (radius: half the diagonal plus a margin) and is as wide as the figure is large; the box
    itself stays a thin dashed line."""

    import cv2

    for number, (x, y, w, h) in enumerate(boxes, 1):
        cx, cy = round((x + w / 2) * mpp0[0] / mpp), round((y + h / 2) * mpp0[1] / mpp)
        radius = round((((w * mpp0[0]) ** 2 + (h * mpp0[1]) ** 2) ** 0.5 / 2) / mpp + 25 * unit)
        cv2.circle(image, (cx, cy), radius, ANNOTATION, max(2, int(2 * unit)))
        label = str(number)
        scale, width = 1.6 * unit, max(2, int(3 * unit))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, width)
        origin = (min(max(cx + radius, 0), image.shape[1] - tw - 1), max(cy - radius, th + 1))   # above right of the ring, inside the figure
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), width + 4 * max(1, int(unit)))
        cv2.putText(image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, ANNOTATION, width)


def write_jpeg(image: np.ndarray, path: Path, quality: int = QUALITY) -> str:
    """Atomic: the file appears whole or not at all."""

    import cv2

    temp = path.with_name(path.stem + ".tmp.jpg")
    try:
        if not cv2.imwrite(str(temp), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
            raise OSError(f"could not write {temp}")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return str(path)


def save_figures(mpp0: tuple[float, float], mpp: float, stages: Mapping[str, Any], boxes: Sequence[Sequence[int]],
                 figures: Mapping[str, Any], name: str, device: Any, annotations: Sequence[Sequence[int]] = ()) -> list[str]:
    """Write the JPEGs whose switch is on; return their paths.

    Optional keys of ``figures`` (validated in ``wsi_processing.validate_figures``): ``downscale`` (integer, the figure is that
    many times smaller than the stages), ``quality`` (JPEG) and ``box_thickness`` (px). Colours are fixed.

    ``stages`` are the arrays of ``segnet.infer_stages(..., with_image=True)`` (at ``mpp`` um/px), ``mpp0`` the level-0 MPP of
    the slide. ``annotations`` are level-0 ``[x, y, w, h]`` ROIs of an annotation, if there is one: they are drawn in green
    on the ``boxes`` figure (nothing is drawn, and no legend added, when there are none). The painting is done on ``device`` in bands of rows, the files are encoded in parallel."""

    from concurrent.futures import ThreadPoolExecutor

    import cv2

    wanted = [stage for stage in STAGES if figures.get(stage)]
    if not wanted:
        return []
    labels, epithelium, slide_image = stages["tissue"], stages["epithelium"], stages["image"]
    rows, cols = epithelium.shape
    step = figures.get("downscale", 1)                        # the figure is ``step`` times smaller than the stages
    out_rows, out_cols = -(-rows // step), -(-cols // step)
    fig_mpp = mpp * step
    images = {stage: np.empty((out_rows, out_cols, 3), dtype=np.uint8) for stage in wanted}
    counts = np.zeros(256, dtype=np.int64)
    band_rows = BAND - BAND % step                            # whole steps per band, so the bands join without a seam
    for r0 in range(0, rows, band_rows):
        r1, e0, e1 = min(rows, r0 + band_rows), max(0, r0 - 4), min(rows, r0 + band_rows + 4)   # the extra rows keep the outline whole
        counts += np.bincount(labels[r0:r1].ravel(), minlength=256)
        for stage in wanted:
            band = paint(stage, slide_image[e0:e1], labels[e0:e1], epithelium[e0:e1], stages["classes"], device)[r0 - e0:r1 - e0]
            if step > 1:
                band = cv2.resize(band, (out_cols, -(-(r1 - r0) // step)), interpolation=cv2.INTER_AREA)
            images[stage][r0 // step:r0 // step + band.shape[0]] = band
    unit = out_cols / 1600                                  # sizes of text and lines are those of a 1600 px wide image
    thickness = figures.get("box_thickness", BOX_THICKNESS)    # thin: the figure is looked at zoomed in
    directory = Path(figures["dir"])
    directory.mkdir(parents=True, exist_ok=True)
    for stage in wanted:
        if stage == "boxes":
            draw_boxes(images[stage], boxes, mpp0, fig_mpp, thickness, BOX)
            draw_boxes(images[stage], annotations, mpp0, fig_mpp, thickness, ANNOTATION)
            mark_annotations(images[stage], annotations, mpp0, fig_mpp, unit)
        if stage == "tissue":
            legend(images[stage], stages["classes"], counts[:len(stages["classes"])] > 0, unit)
        caption(images[stage], f"{name} {stage}" + (f" ({len(boxes)}" + (f", annotation {len(annotations)}" if len(annotations) else "") + ")" if stage == "boxes" else ""), unit)
    quality = figures.get("quality", QUALITY)
    with ThreadPoolExecutor(len(wanted)) as pool:
        return list(pool.map(lambda stage: write_jpeg(images[stage], directory / f"{name}_{stage}.jpg", quality), wanted))
