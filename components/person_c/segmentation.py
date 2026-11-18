"""The student network on a whole-slide image, and pictures of what it found (mode ``region_proposal``).

A DeepLab v3+ student trained without annotation (PIPELINE/SEGNET, ``Unsup``, pseudo labels of the SCLIP teacher on BRACS
slides, class set c9) labels the slide; the union of ``segnet.roi_classes`` (TUM, BEN) is returned as a boolean mask at
``postprocess_mpp`` (1 um/px), from which ``interest_pattern`` makes the ROI boxes. The file has four blocks:

1. Network     the DeepLab v3+ definition (inference copy of SEGNET's ``SegNet``; the module names must match the checkpoint)
2. Student     checkpoint check and loading, window prediction with cosine-squared blending
3. Slide       tiles of the slide read by worker processes, the stages: class map, epithelium mask
4. Figures     one JPEG per WSI and stage: tissue, epithelium, boxes
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import openslide
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from contracts.paths import resolve_person_reference_path

UNLABELED = 255          # initial value of the label map; every pixel of a slide is covered by a tile


# ============================================================ 1. Network
RATES = (12, 24, 36)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
OUT_STRIDE = 4


def conv(i: int, o: int, k: int = 3, d: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, k, padding=d * (k // 2), dilation=d, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    )


class ASPP(nn.Module):
    def __init__(self, i: int = 2048, o: int = 256) -> None:
        super().__init__()
        self.branches = nn.ModuleList([conv(i, o, 1)] + [conv(i, o, 3, d) for d in RATES])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(i, o, 1, bias=False), nn.ReLU(inplace=True))
        self.project = nn.Sequential(conv(o * (len(RATES) + 2), o, 1), nn.Dropout(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.project(torch.cat([b(x) for b in self.branches] + [pooled], 1))


def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class SegNet(nn.Module):
    out_stride = OUT_STRIDE

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        from torchvision.models import resnet101

        r = resnet101(weights=None, replace_stride_with_dilation=[False, True, True])
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.body1 = nn.Sequential(r.maxpool, r.layer1)
        self.body2 = nn.Sequential(r.layer2, r.layer3, r.layer4)
        self.aspp = ASPP()
        self.low4 = conv(256, 48, 1)
        self.dec4 = nn.Sequential(conv(256 + 48, 256), conv(256, 256))
        self.head = nn.Conv2d(256, n_classes, 1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        x = (x01 - self.mean) / self.std
        c4 = self.body1(self.stem(x))
        y = self.aspp(self.body2(c4))
        return self.head(self.dec4(torch.cat([_up(y, c4), self.low4(c4)], 1)))


# ============================================================ 2. Student
CELL_STRIDE = 4          # output stride of the student: a cell is CELL_STRIDE x mpp um wide


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every hole of the mask, whatever its size: what lies inside a ring belongs to the ring. A hole is a part of the
    background (4-connectivity) that does not reach the border of the image."""


    img = np.pad(mask.astype(np.uint8), 1)
    cv2.floodFill(img, None, (0, 0), 2, flags=4)               # the background that reaches the border
    return img[1:-1, 1:-1] != 2


def window_starts(n: int, win: int, step: int) -> list[int]:
    xs = list(range(0, max(n - win, 0) + 1, step))
    if xs[-1] != max(n - win, 0):
        xs.append(max(n - win, 0))
    return xs


def load_model(settings: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    """Resolve and verify the checkpoint, load it -> ({"net", "classes", "window", "mpp"}, device).

    ``segnet.checkpoint`` is ``{"path": below reference/person_c, "revision": "sha256:<digest>"}``.
    """


    cfg = settings["segnet"]
    if not torch.cuda.is_available():
        raise RuntimeError("person_c needs a CUDA GPU and none is available")
    device = torch.device("cuda")
    entry = cfg["checkpoint"]
    path = resolve_person_reference_path("person_c", entry["path"])
    if not path.is_file():
        raise FileNotFoundError(f"person_c SEGNET checkpoint is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if f"sha256:{digest.hexdigest()}" != entry["revision"]:
        raise ValueError(f"person_c SEGNET checkpoint checksum mismatch: {path}")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    classes, window, mpp = list(ck["classes"]), int(ck["config"]["window_px"]), float(ck["config"]["mpp"])
    net = SegNet(len(classes))
    result = net.load_state_dict(ck["state"], strict=False)
    if result.unexpected_keys or result.missing_keys:
        raise ValueError(f"person_c SEGNET checkpoint {path} does not fit the network: {result}")
    missing = [c for c in cfg["roi_classes"] if c not in classes]
    if missing:
        raise ValueError(f"person_c segnet.roi_classes {missing} are not classes of the checkpoint {classes}")
    return {"net": net.to(device).eval(), "classes": classes, "window": window, "mpp": mpp}, device


def predict_cells(img: np.ndarray, net: Any, window: int, device: Any, batch: int) -> Any:
    """uint8 (H, W, 3) at the student's mpp -> class probabilities (K, ceil(H/4), ceil(W/4)) on the device.

    The same rule as ``Predictor._probs`` of SEGNET: windows of ``window`` px at half overlap blended with a
    cosine-squared weight; an image smaller than
    the window is one window of its own size.
    """


    s = CELL_STRIDE
    H, W = img.shape[:2]
    size = min(window, H // s * s, W // s * s)
    Hp, Wp = max(-(-H // s) * s, size), max(-(-W // s) * s, size)
    g = size // s
    k_classes = None
    with torch.inference_mode():
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].to(device).float() / 255
        x = F.pad(x, (0, Wp - W, 0, Hp - H), mode="replicate")
        w1 = torch.sin(math.pi * (torch.arange(g, device=device) + 0.5) / g) ** 2 + 0.02
        w = w1[:, None] * w1[None]
        acc = wsum = None
        coords = [(y, x0) for y in window_starts(Hp, size, size // 2) for x0 in window_starts(Wp, size, size // 2)]
        for i in range(0, len(coords), batch):
            chunk = coords[i:i + batch]
            wins = torch.cat([x[:, :, y:y + size, x0:x0 + size] for y, x0 in chunk])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                p = torch.softmax(net(wins).float(), 1)
            if acc is None:
                k_classes = p.shape[1]
                acc = torch.zeros(k_classes, Hp // s, Wp // s, device=device)
                wsum = torch.zeros(1, Hp // s, Wp // s, device=device)
            for pi, (y, x0) in zip(p, chunk):
                acc[:, y // s:y // s + g, x0 // s:x0 // s + g] += pi * w
                wsum[:, y // s:y // s + g, x0 // s:x0 // s + g] += w
        return (acc / wsum.clamp_min(1e-8))[:, :-(-H // s), :-(-W // s)]


# ============================================================ 3. Slide
def read_level(slide: Any, mpp: tuple[float, float], target_mpp: float) -> int:
    candidates = [
        level for level, downsample in enumerate(slide.level_downsamples)
        if all(axis * float(downsample) <= target_mpp * 1.01 for axis in mpp)
    ]
    return max(candidates) if candidates else 0


def read_um(slide: Any, mpp0: tuple[float, float], x_um: int, y_um: int, w_um: int, h_um: int, mpp: float) -> np.ndarray:
    """The region [x_um, x_um + w_um) x [y_um, y_um + h_um) of the slide as uint8 RGB at ``mpp`` um/px."""


    level = read_level(slide, mpp0, mpp)          # the coarsest level that is not coarser than ``mpp`` (1 % tolerance)
    ds_level = slide.level_downsamples[level]
    x0, y0 = round(x_um / mpp0[0]), round(y_um / mpp0[1])
    w_l, h_l = max(1, round(w_um / mpp0[0] / ds_level)), max(1, round(h_um / mpp0[1] / ds_level))
    rgba = np.asarray(slide.read_region((x0, y0), level, (w_l, h_l)))
    alpha = rgba[..., 3:4].astype(np.float32) / 255
    rgb = (rgba[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return cv2.resize(rgb, (round(w_um / mpp), round(h_um / mpp)), interpolation=cv2.INTER_AREA)


UNLABELED = 255          # initial value of the label map; every pixel of a slide is covered by a tile
TILE_UM, MARGIN_UM = 1792, 112      # the slide is read in tiles of TILE_UM at the student's mpp, with MARGIN_UM of context around
BATCH_SIZE = 32                     # windows per forward pass (about 1 GiB of GPU memory)
NUM_WORKERS = 6                     # processes that read the tiles while the GPU predicts the previous ones


class TileReader:
    """The tiles of a slide for ``torch.utils.data.DataLoader``: item i is ``(x0, y0, image)``, the tile with its
    context at the student's mpp. Every worker opens its own slide (a slide cannot be shared between processes)."""

    def __init__(self, path: str, mpp0: tuple[float, float], origins: list[tuple[int, int]], boxes: list[tuple[int, int, int, int]], mpp: float) -> None:
        self.path, self.mpp0, self.origins, self.boxes, self.mpp, self.slide = path, mpp0, origins, boxes, mpp, None

    def __len__(self) -> int:
        return len(self.boxes)

    def __getitem__(self, i: int) -> tuple[int, int, np.ndarray]:
        
        if self.slide is None:
            cv2.setNumThreads(1)                       # the workers share the CPUs: one thread each
            self.slide = openslide.OpenSlide(self.path)
        x0, y0, x1, y1 = self.boxes[i]
        return x0, y0, read_um(self.slide, self.mpp0, x0, y0, x1 - x0, y1 - y0, self.mpp)


def infer_stages(path: str, slide: Any, mpp0: tuple[float, float], settings: Mapping[str, Any], model: Mapping[str, Any], device: Any,
                 with_image: bool = False) -> dict[str, Any]:
    """The stages at ``postprocess_mpp``:
    ``tissue`` (stage 1: the multi-class tissue segmentation, the index of the class of every pixel in ``classes``;
    the student itself labels the glass, there is no foreground step) and ``epithelium`` (stage 2: the union of
    ``segnet.roi_classes``, holes filled). The boxes (stage 3) are made from ``epithelium`` by the caller.
    With ``with_image`` the slide at ``postprocess_mpp`` is returned as ``image`` (RGB) for the figures; the tiles are
    read for the student anyway, so this costs no second read.

    The class probabilities of the cells (4 um) are enlarged bilinearly to ``postprocess_mpp`` before the argmax, so
    the borders are smooth. Everything that is a per-pixel operation (enlarging, argmax, the label map, the union of the
    classes) stays on the GPU; the tiles are read by ``NUM_WORKERS`` processes while the GPU predicts the previous ones."""


    cfg = settings["segnet"]
    classes, mpp = model["classes"], model["mpp"]
    tile, margin = TILE_UM, MARGIN_UM
    w0, h0 = slide.dimensions
    w_um, h_um = math.ceil(w0 * mpp0[0]), math.ceil(h0 * mpp0[1])
    target = float(settings["postprocess_mpp"])
    scale = CELL_STRIDE * mpp / target
    origins = [(tx, ty) for ty in range(0, h_um, tile) for tx in range(0, w_um, tile)]
    boxes = [(max(tx - margin, 0), max(ty - margin, 0), min(tx + tile + margin, w_um), min(ty + tile + margin, h_um)) for tx, ty in origins]
    loader = DataLoader(TileReader(path, mpp0, origins, boxes, mpp), batch_size=None, shuffle=False, num_workers=NUM_WORKERS,
                        prefetch_factor=2)
    shape = (math.ceil(h0 * mpp0[1] / target), math.ceil(w0 * mpp0[0] / target))
    tissue = torch.full(shape, UNLABELED, dtype=torch.uint8, device=device)
    image = np.full((*shape, 3), 255, dtype=np.uint8) if with_image else None
    side = round(tile / target)
    for (tx, ty), (x0, y0, img) in zip(origins, loader):
        probs = predict_cells(img.numpy(), model["net"], model["window"], device, BATCH_SIZE)
        label = F.interpolate(probs[None], scale_factor=scale, mode="bilinear", align_corners=False)[0].argmax(0).to(torch.uint8)
        cx, cy = round((tx - int(x0)) / target), round((ty - int(y0)) / target)
        part = label[cy:cy + side, cx:cx + side]
        gx, gy = round(tx / target), round(ty / target)
        h, w = tissue[gy:gy + part.shape[0], gx:gx + part.shape[1]].shape
        tissue[gy:gy + h, gx:gx + w] = part[:h, :w]
        if with_image:
            ix, iy, iside = round((tx - int(x0)) / mpp), round((ty - int(y0)) / mpp), round(tile / mpp)
            crop = img.numpy()[iy:iy + iside, ix:ix + iside]
            if mpp != target:
                crop = cv2.resize(crop, (part.shape[1], part.shape[0]), interpolation=cv2.INTER_AREA)
            image[gy:gy + h, gx:gx + w] = crop[:h, :w]
    epithelium = torch.zeros_like(tissue, dtype=torch.bool)
    for name in cfg["roi_classes"]:
        epithelium |= tissue == classes.index(name)
    out = {"tissue": tissue.cpu().numpy(), "epithelium": fill_holes(epithelium.cpu().numpy()), "classes": classes}
    if with_image:
        out["image"] = image
    return out


# ============================================================ 4. Figures
STAGES = ("tissue", "epithelium", "boxes")
EPITHELIUM = (0, 70, 255)                    # RGB
# The colours of the boxes lie outside the hues of H&E (pink, purple, blue, white): amber for the boxes of the method, lime for
# an annotation. BGR: the images of this module are BGR from ``paint`` on.
BOX = (0, 176, 255)
ANNOTATION = (79, 255, 124)
LINE_ALPHA, EDGE_ALPHA = 0.7, 0.5            # opacity of a box line and of its dark edge: the slide shows through both
# colours of the classes (c9 of the project's DATA/meta/classes.json); a class that is not listed is grey
CLASS_COLORS = {"TUM": (215, 35, 55), "BEN": (240, 120, 30), "STR": (15, 155, 65), "LYM": (35, 95, 215), "NEC": (240, 180, 10),
                "FAT": (40, 190, 230), "VES": (130, 60, 190), "BLD": (110, 20, 45), "SEC": (200, 50, 160), "OTH": (25, 30, 75)}
BOX_THICKNESS = 2                            # px of the figure (a dark edge of 1 px is added on each side)
BAND = 1024                                  # rows per band
QUALITY = 90                                 # JPEG quality: clear to the eye, a tenth of the size of a PNG
ALPHA = {"tissue": 0.6, "epithelium": 0.55, "boxes": 0.55}      # the boxes figure has the look of the epithelium figure


def paint(stage: str, band: np.ndarray, labels: np.ndarray, epithelium: np.ndarray, classes: Sequence[str], device: Any) -> np.ndarray:
    """The stage drawn on a band of the slide (uint8 RGB, rows of the stage arrays) -> the painted band, as BGR. On the GPU."""


    image = torch.from_numpy(band).to(device)
    epi = torch.from_numpy(epithelium).to(device)
    if stage == "tissue":
        lab = torch.from_numpy(labels).to(device)
        lut = torch.zeros((256, 3), dtype=torch.float32, device=device)
        for index, name in enumerate(classes):
            lut[index] = torch.tensor(CLASS_COLORS.get(name, (128, 128, 128)), dtype=torch.float32)
        color, mask = lut[lab.long()], lab != UNLABELED
        lf = lab[None, None].float()                              # a black line of about 3 px between two classes
        edge = (F.max_pool2d(lf, 3, 1, 1) != -F.max_pool2d(-lf, 3, 1, 1))[0, 0] & mask
    else:
        color, mask = torch.tensor(EPITHELIUM, dtype=torch.float32, device=device), epi
    alpha = ALPHA[stage]
    mixed = ((1 - alpha) * image.float() + alpha * color).to(torch.uint8)
    image = torch.where(mask[..., None], mixed, image)
    if stage == "tissue":
        image = torch.where(edge[..., None], torch.zeros_like(image), image)
    else:                                                       # a black line of about 3 px along the border of the mask
        inner = -F.max_pool2d(-epi[None, None].float(), 7, 1, 3)[0, 0] > 0.5
        image = torch.where((epi & ~inner)[..., None], torch.zeros_like(image), image)
    return image.flip(-1).cpu().numpy()                        # BGR, the order OpenCV draws and writes in


def legend(image: np.ndarray, classes: Sequence[str], present: Sequence[bool], unit: float) -> None:

    names = [name for name, here in zip(classes, present) if here]
    for row, name in enumerate(names):
        y, color = int((60 + row * 34) * unit), tuple(CLASS_COLORS.get(name, (128, 128, 128)))[::-1]
        cv2.rectangle(image, (int(12 * unit), y), (int(40 * unit), y + int(24 * unit)), color, -1)
        cv2.rectangle(image, (int(12 * unit), y), (int(40 * unit), y + int(24 * unit)), (0, 0, 0), max(1, int(unit)))
        cv2.putText(image, name, (int(50 * unit), y + int(20 * unit)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * unit, (0, 0, 0), max(1, int(2 * unit)))


def caption(image: np.ndarray, text: str, unit: float) -> None:

    for color, width in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(image, text, (int(12 * unit), int(32 * unit)), cv2.FONT_HERSHEY_SIMPLEX, unit, color, max(1, int(width * unit)))


def dashed_line(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int], thickness: int, pattern: tuple[float, float]) -> None:
    """In place: a dashed line on a one-channel ``mask`` (dash and gap in line widths, ``pattern``)."""


    length = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    on, off = max(2, round(pattern[0] * thickness)), max(2, round(pattern[1] * thickness))
    for t in range(0, length, on + off):
        a, b = t / length, min(t + on, length) / length
        p, q = ((round(start[0] + (end[0] - start[0]) * u), round(start[1] + (end[1] - start[1]) * u)) for u in (a, b))
        cv2.line(mask, p, q, 255, thickness)


def blend_box(image: np.ndarray, box: Sequence[int], color: tuple[int, int, int], thickness: int, pattern: tuple[float, float]) -> None:
    """In place: the dashed rectangle ``box`` (pixels ``x0, y0, x1, y1``) laid over the slide as a translucent colour, with a
    translucent dark edge of 1 px on each side, so that the slide shows through the line as well as between the dashes. Only the
    part of the image around the box is touched."""

    x0, y0, x1, y1 = box
    pad = thickness + 2
    left, top = max(x0 - pad, 0), max(y0 - pad, 0)
    right, bottom = min(x1 + pad + 1, image.shape[1]), min(y1 + pad + 1, image.shape[0])
    if right <= left or bottom <= top:
        return
    core = np.zeros((bottom - top, right - left), dtype=np.uint8)
    halo = np.zeros_like(core)
    corners = ((x0 - left, y0 - top), (x1 - left, y0 - top), (x1 - left, y1 - top), (x0 - left, y1 - top))
    for start, end in zip(corners, corners[1:] + corners[:1]):
        dashed_line(halo, start, end, thickness + 2, pattern)
        dashed_line(core, start, end, thickness, pattern)
    part = image[top:bottom, left:right]
    dark = (halo > 0) & (core == 0)
    part[dark] = (part[dark] * (1 - EDGE_ALPHA)).astype(np.uint8)
    line = core > 0
    part[line] = (part[line] * (1 - LINE_ALPHA) + np.array(color, dtype=np.float32) * LINE_ALPHA).astype(np.uint8)


def tag(image: np.ndarray, text: str, corner: tuple[int, int], color: tuple[int, int, int]) -> None:
    """In place: a small number on a dark label just above the upper left ``corner`` (below it at the top of the figure).
    The size is the same for every box."""


    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x = min(max(corner[0], 0), max(image.shape[1] - width - 7, 0))
    y = corner[1] - 3
    if y - height - 4 < 0:
        y = corner[1] + height + 7
    cv2.rectangle(image, (x, y - height - 4), (x + width + 6, y + 2), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 3, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_boxes(image: np.ndarray, boxes: Sequence[Sequence[int]], mpp0: tuple[float, float], mpp: float, thickness: int,
               annotation: bool = False) -> None:
    """In place: level-0 ``[x, y, w, h]`` boxes on an image of ``mpp`` um/px, each with its number. The boxes of the method are
    amber, short-dashed and numbered from 0 (``local_idx`` of the ROI in E); an annotation is lime, one pixel wider, with long
    dashes, numbered from 1. The two sets differ in colour and in rhythm, so that they stay apart where they overlap."""

    color, width, pattern = (ANNOTATION, thickness + 1, (6.0, 3.0)) if annotation else (BOX, thickness, (3.5, 2.5))
    for number, (x, y, w, h) in enumerate(boxes, 1 if annotation else 0):
        x0, y0 = round(x * mpp0[0] / mpp), round(y * mpp0[1] / mpp)
        x1, y1 = round((x + w) * mpp0[0] / mpp), round((y + h) * mpp0[1] / mpp)
        blend_box(image, (x0, y0, x1, y1), color, width, pattern)
        tag(image, str(number), (x0, y0), color)


def write_jpeg(image: np.ndarray, path: Path) -> str:
    """Atomic: the file appears whole or not at all."""


    temp = path.with_name(path.stem + ".tmp.jpg")
    try:
        if not cv2.imwrite(str(temp), image, [cv2.IMWRITE_JPEG_QUALITY, QUALITY]):
            raise OSError(f"could not write {temp}")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return str(path)


def save_figures(mpp0: tuple[float, float], mpp: float, stages: Mapping[str, Any], boxes: Sequence[Sequence[int]],
                 figures: Mapping[str, Any], name: str, device: Any, annotations: Sequence[Sequence[int]] = ()) -> list[str]:
    """Write the JPEGs whose switch is on; return their paths.

    Optional key of ``figures`` (validated in ``interest_pattern.validate_figures``): ``downscale`` (integer, the figure is that
    many times smaller than the stages).

    ``stages`` are the arrays of ``segnet.infer_stages(..., with_image=True)`` (at ``mpp`` um/px), ``mpp0`` the level-0 MPP of
    the slide. ``annotations`` are level-0 ``[x, y, w, h]`` ROIs of an annotation, if there is one: they are drawn in green
    on the ``boxes`` figure (nothing is drawn, and no legend added, when there are none). The painting is done on ``device`` in bands of rows, the files are encoded in parallel."""

    from concurrent.futures import ThreadPoolExecutor


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
    thickness = BOX_THICKNESS
    directory = Path(figures["dir"])
    directory.mkdir(parents=True, exist_ok=True)
    for stage in wanted:
        if stage == "boxes":
            draw_boxes(images[stage], boxes, mpp0, fig_mpp, thickness)
            draw_boxes(images[stage], annotations, mpp0, fig_mpp, thickness, annotation=True)
        if stage == "tissue":
            legend(images[stage], stages["classes"], counts[:len(stages["classes"])] > 0, unit)
        caption(images[stage], f"{name} {stage}" + (f" ({len(boxes)}" + (f", annotation {len(annotations)}" if len(annotations) else "") + ")" if stage == "boxes" else ""), unit)
    with ThreadPoolExecutor(len(wanted)) as pool:
        return list(pool.map(lambda stage: write_jpeg(images[stage], directory / f"{name}_{stage}.jpg"), wanted))
