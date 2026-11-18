"""Semantic mask of one WSI from the SEGNET student (backend ``segnet``).

A DeepLab student trained without annotation (PIPELINE/SEGNET, ``Unsup``, pseudo labels of the SCLIP teacher on
BRACS slides, class set c9) labels the slide; the union of ``roi_classes`` (TUM, BEN) is returned as a boolean
mask at ``postprocess_mpp`` (1 um/px), which ``wsi_processing.level0_boxes`` turns into ROI boxes.

* The slide is read at the student's 1 um/px in tiles of ``tile_um`` with ``margin_um`` of context on every side;
  the student's windows (224 px, half overlap, cosine-squared blending, ``tta_views`` dihedral views, mean over
  the checkpoints) are those of SEGNET's ``Predictor``.
* The student outputs one probability vector per 4 um cell; it is enlarged bilinearly to ``postprocess_mpp`` and the label of a pixel is the class of the highest probability (argmax over c9).
* Heavy imports (torch, openslide, cv2) happen inside the functions so contract tests stay light.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

CELL_STRIDE = 4          # output stride of the student: a cell is CELL_STRIDE x mpp um wide
VIEWS = tuple((k, f) for f in (0, 1) for k in range(4))   # the eight dihedral views of SEGNET's TTA


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every hole of the mask, whatever its size: what lies inside a ring belongs to the ring. A hole is a part of the
    background (4-connectivity) that does not reach the border of the image."""

    import cv2

    img = np.pad(mask.astype(np.uint8), 1)
    cv2.floodFill(img, None, (0, 0), 2, flags=4)               # the background that reaches the border
    return img[1:-1, 1:-1] != 2


def window_starts(n: int, win: int, step: int) -> list[int]:
    xs = list(range(0, max(n - win, 0) + 1, step))
    if xs[-1] != max(n - win, 0):
        xs.append(max(n - win, 0))
    return xs


def load_model(settings: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    """Resolve and verify the checkpoints, load them -> ({"nets", "classes", "window", "mpp"}, device).

    Every ``segnet.checkpoints`` entry is ``{"path": below reference/person_c, "revision": "sha256:<digest>"}``.
    """

    import torch

    from components.person_c.region.segnet_net import SegNet
    from contracts.paths import resolve_person_reference_path

    cfg = settings["segnet"]
    device_name = settings["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("person_c CUDA was requested but is unavailable")
    device = torch.device(device_name)
    nets, classes, window, mpp = [], None, None, None
    for entry in cfg["checkpoints"]:
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
        if classes is None:
            classes, window, mpp = list(ck["classes"]), int(ck["config"]["window_px"]), float(ck["config"]["mpp"])
        elif list(ck["classes"]) != classes:
            raise ValueError(f"person_c SEGNET checkpoint {path} has other classes than the first one")
        net = SegNet(len(classes))
        result = net.load_state_dict(ck["state"], strict=False)
        if result.unexpected_keys or result.missing_keys:
            raise ValueError(f"person_c SEGNET checkpoint {path} does not fit the network: {result}")
        nets.append(net.to(device).eval())
    missing = [c for c in cfg["roi_classes"] if c not in classes]
    if missing:
        raise ValueError(f"person_c segnet.roi_classes {missing} are not classes of the checkpoints {classes}")
    return {"nets": nets, "classes": classes, "window": window, "mpp": mpp}, device


def predict_cells(img: np.ndarray, nets: Sequence[Any], window: int, device: Any, views: int, batch: int) -> Any:
    """uint8 (H, W, 3) at the student's mpp -> class probabilities (K, ceil(H/4), ceil(W/4)) on the device.

    The same rule as ``Predictor._probs`` of SEGNET: windows of ``window`` px at half overlap blended with a
    cosine-squared weight, the mean of the dihedral views, the mean of the networks; an image smaller than
    the window is one window of its own size.
    """

    import torch
    import torch.nn.functional as F

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
            p = 0
            for k, f in VIEWS[:views]:
                v = torch.rot90(wins, k, (2, 3))
                v = v.flip(3) if f else v
                for net in nets:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        o = torch.softmax(net(v).float(), 1)
                    o = o.flip(3) if f else o
                    p = p + torch.rot90(o, -k, (2, 3))
            p = p / (views * len(nets))
            if acc is None:
                k_classes = p.shape[1]
                acc = torch.zeros(k_classes, Hp // s, Wp // s, device=device)
                wsum = torch.zeros(1, Hp // s, Wp // s, device=device)
            for pi, (y, x0) in zip(p, chunk):
                acc[:, y // s:y // s + g, x0 // s:x0 // s + g] += pi * w
                wsum[:, y // s:y // s + g, x0 // s:x0 // s + g] += w
        return (acc / wsum.clamp_min(1e-8))[:, :-(-H // s), :-(-W // s)]


def read_um(slide: Any, mpp0: tuple[float, float], x_um: int, y_um: int, w_um: int, h_um: int, mpp: float) -> np.ndarray:
    """The region [x_um, x_um + w_um) x [y_um, y_um + h_um) of the slide as uint8 RGB at ``mpp`` um/px."""

    import cv2

    from components.person_c import wsi_processing as wsi

    level = wsi.read_level(slide, mpp0, mpp)          # the coarsest level that is not coarser than ``mpp`` (1 % tolerance)
    ds_level = slide.level_downsamples[level]
    x0, y0 = round(x_um / mpp0[0]), round(y_um / mpp0[1])
    w_l, h_l = max(1, round(w_um / mpp0[0] / ds_level)), max(1, round(h_um / mpp0[1] / ds_level))
    rgba = np.asarray(slide.read_region((x0, y0), level, (w_l, h_l)))
    alpha = rgba[..., 3:4].astype(np.float32) / 255
    rgb = (rgba[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return cv2.resize(rgb, (round(w_um / mpp), round(h_um / mpp)), interpolation=cv2.INTER_AREA)


UNLABELED = 255          # a cell that no tile covered (never happens for a whole slide)


class TileReader:
    """The tiles of a slide for ``torch.utils.data.DataLoader``: item i is ``(x0, y0, image)``, the tile with its
    context at the student's mpp. Every worker opens its own slide (a slide cannot be shared between processes)."""

    def __init__(self, path: str, mpp0: tuple[float, float], origins: list[tuple[int, int]], boxes: list[tuple[int, int, int, int]], mpp: float) -> None:
        self.path, self.mpp0, self.origins, self.boxes, self.mpp, self.slide = path, mpp0, origins, boxes, mpp, None

    def __len__(self) -> int:
        return len(self.boxes)

    def __getitem__(self, i: int) -> tuple[int, int, np.ndarray]:
        import cv2
        import openslide

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
    classes) stays on the GPU; the tiles are read by ``segnet.num_workers`` processes while the GPU predicts the previous ones."""

    import cv2
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    cfg = settings["segnet"]
    classes, mpp = model["classes"], model["mpp"]
    tile, margin = cfg["tile_um"], cfg["margin_um"]
    w0, h0 = slide.dimensions
    w_um, h_um = math.ceil(w0 * mpp0[0]), math.ceil(h0 * mpp0[1])
    target = float(settings["postprocess_mpp"])
    scale = CELL_STRIDE * mpp / target
    origins = [(tx, ty) for ty in range(0, h_um, tile) for tx in range(0, w_um, tile)]
    boxes = [(max(tx - margin, 0), max(ty - margin, 0), min(tx + tile + margin, w_um), min(ty + tile + margin, h_um)) for tx, ty in origins]
    workers = cfg["num_workers"]
    loader = DataLoader(TileReader(path, mpp0, origins, boxes, mpp), batch_size=None, shuffle=False, num_workers=workers,
                        prefetch_factor=2 if workers else None)
    shape = (math.ceil(h0 * mpp0[1] / target), math.ceil(w0 * mpp0[0] / target))
    tissue = torch.full(shape, UNLABELED, dtype=torch.uint8, device=device)
    image = np.full((*shape, 3), 255, dtype=np.uint8) if with_image else None
    side = round(tile / target)
    for (tx, ty), (x0, y0, img) in zip(origins, loader):
        probs = predict_cells(img.numpy(), model["nets"], model["window"], device, cfg["tta_views"], cfg["batch_size"])
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
