"""One JPEG per WSI and per stage, for whoever follows the segmentation (switches: ``figures`` in the config).

Stages: ``tissue`` (the multi-class tissue segmentation, one colour per class with a legend), ``epithelium`` (the processed
segmentation the boxes are made from) and ``boxes`` (the level-0 boxes of the ROIs over the epithelium). Each is the slide at
the resolution of the segmentation (``postprocess_mpp``, 1 um/px) with the stage drawn over it, written to ``figures.dir``
below ``run/output`` as ``<case_id>_<stain_id>_<stage>.jpg``. The slide image is the one the student already read; the
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
        y, color = int((60 + row * 34) * unit), CLASS_COLORS.get(name, (128, 128, 128))[::-1]
        cv2.rectangle(image, (int(12 * unit), y), (int(40 * unit), y + int(24 * unit)), color, -1)
        cv2.rectangle(image, (int(12 * unit), y), (int(40 * unit), y + int(24 * unit)), (0, 0, 0), max(1, int(unit)))
        cv2.putText(image, name, (int(50 * unit), y + int(20 * unit)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * unit, (0, 0, 0), max(1, int(2 * unit)))


def caption(image: np.ndarray, text: str, unit: float) -> None:
    import cv2

    for color, width in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(image, text, (int(12 * unit), int(32 * unit)), cv2.FONT_HERSHEY_SIMPLEX, unit, color, max(1, int(width * unit)))


def draw_boxes(image: np.ndarray, boxes: Sequence[Sequence[int]], mpp0: tuple[float, float], mpp: float, thickness: int,
               color: tuple[int, int, int] = BOX) -> None:
    """In place: level-0 ``[x, y, w, h]`` boxes on an image of ``mpp`` um/px."""

    import cv2

    for x, y, w, h in boxes:
        cv2.rectangle(image, (round(x * mpp0[0] / mpp), round(y * mpp0[1] / mpp)),
                      (round((x + w) * mpp0[0] / mpp), round((y + h) * mpp0[1] / mpp)), color, thickness)


def write_jpeg(image: np.ndarray, path: Path) -> str:
    """Atomic: the file appears whole or not at all."""

    import cv2

    temp = path.with_name(path.stem + ".tmp.jpg")
    try:
        if not cv2.imwrite(str(temp), image, [cv2.IMWRITE_JPEG_QUALITY, QUALITY]):
            raise OSError(f"could not write {temp}")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return str(path)


def save_figures(mpp0: tuple[float, float], mpp: float, stages: Mapping[str, Any], boxes: Sequence[Sequence[int]],
                 figures: Mapping[str, Any], case_id: str, stain_id: str, device: Any, annotations: Sequence[Sequence[int]] = ()) -> list[str]:
    """Write the JPEGs whose switch is on; return their paths.

    ``stages`` are the arrays of ``segnet.infer_stages(..., with_image=True)`` (at ``mpp`` um/px), ``mpp0`` the level-0 MPP of
    the slide. ``annotations`` are level-0 ``[x, y, w, h]`` ROIs of an annotation, if there is one: they are drawn in green
    on the ``boxes`` figure (nothing is drawn, and no legend added, when there are none). The painting is done on ``device`` in bands of rows, the files are encoded in parallel."""

    from concurrent.futures import ThreadPoolExecutor

    wanted = [stage for stage in STAGES if figures.get(stage)]
    if not wanted:
        return []
    labels, epithelium, slide_image = stages["tissue"], stages["epithelium"], stages["image"]
    rows, cols = epithelium.shape
    images = {stage: np.empty((rows, cols, 3), dtype=np.uint8) for stage in wanted}
    counts = np.zeros(256, dtype=np.int64)
    for r0 in range(0, rows, BAND):
        r1, e0, e1 = min(rows, r0 + BAND), max(0, r0 - 4), min(rows, r0 + BAND + 4)       # the extra rows keep the outline whole
        counts += np.bincount(labels[r0:r1].ravel(), minlength=256)
        for stage in wanted:
            band = paint(stage, slide_image[e0:e1], labels[e0:e1], epithelium[e0:e1], stages["classes"], device)
            images[stage][r0:r1] = band[r0 - e0:r1 - e0]
    unit = cols / 1600                                      # sizes of text and lines are those of a 1600 px wide image
    directory = Path(figures["dir"])
    directory.mkdir(parents=True, exist_ok=True)
    for stage in wanted:
        if stage == "boxes":
            draw_boxes(images[stage], boxes, mpp0, mpp, max(2, int(2 * unit)))
            draw_boxes(images[stage], annotations, mpp0, mpp, max(2, int(2 * unit)), ANNOTATION)
        if stage == "tissue":
            legend(images[stage], stages["classes"], counts[:len(stages["classes"])] > 0, unit)
        caption(images[stage], f"{stain_id} {stage}" + (f" ({len(boxes)}" + (f", annotation {len(annotations)}" if len(annotations) else "") + ")" if stage == "boxes" else ""), unit)
    with ThreadPoolExecutor(len(wanted)) as pool:
        return list(pool.map(lambda stage: write_jpeg(images[stage], directory / f"{case_id}_{stain_id}_{stage}.jpg"), wanted))
