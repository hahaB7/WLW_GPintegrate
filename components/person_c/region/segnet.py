"""Semantic mask of one WSI from the SEGNET student (backend ``segnet``), the counterpart of ``wsi_processing.infer_mask``.

A DeepLab student trained without annotation (PIPELINE/SEGNET, ``Unsup``, pseudo labels of the SCLIP teacher on
BRACS slides, class set c9) labels the slide; the union of ``roi_classes`` (TUM, BEN) is returned as a boolean
mask at ``postprocess_mpp``, which ``wsi_processing.level0_boxes`` turns into ROI boxes as for MRCPS.

* The slide is read at the student's 1 um/px in tiles of ``tile_um`` with ``margin_um`` of context on every side;
  the student's windows (224 px, half overlap, cosine-squared blending, ``tta_views`` dihedral views, mean over
  the checkpoints) are those of SEGNET's ``Predictor``.
* The label of a cell (4 um, the output stride) is the class of the highest probability (argmax over c9).
* Heavy imports (torch, openslide, cv2) happen inside the functions so contract tests stay light.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

CELL_STRIDE = 4          # output stride of the student: a cell is CELL_STRIDE x mpp um wide
VIEWS = tuple((k, f) for f in (0, 1) for k in range(4))   # the eight dihedral views of SEGNET's TTA


def resample_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Boolean cell mask -> boolean mask of ``shape`` (rows, cols): a pixel is set when most of its area is set."""

    import cv2

    return cv2.resize(mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_AREA) > 0.5


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every hole of the mask, whatever its size: what lies inside a ring belongs to the ring."""

    from scipy import ndimage

    return ndimage.binary_fill_holes(mask)


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

    ds = mpp / mpp0[0]
    level = slide.get_best_level_for_downsample(ds)
    ds_level = slide.level_downsamples[level]
    x0, y0 = round(x_um / mpp0[0]), round(y_um / mpp0[1])
    w_l, h_l = max(1, round(w_um / mpp0[0] / ds_level)), max(1, round(h_um / mpp0[1] / ds_level))
    rgba = np.asarray(slide.read_region((x0, y0), level, (w_l, h_l)))
    alpha = rgba[..., 3:4].astype(np.float32) / 255
    rgb = (rgba[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return cv2.resize(rgb, (round(w_um / mpp), round(h_um / mpp)), interpolation=cv2.INTER_AREA)


def tissue_fraction(img: np.ndarray) -> float:
    """Share of pixels whose channels differ by more than 20 (glass is grey / white)."""

    return float((np.ptp(img.astype(np.int16), axis=2) > 20).mean())


def infer_mask(slide: Any, mpp0: tuple[float, float], settings: Mapping[str, Any], model: Mapping[str, Any], device: Any) -> np.ndarray:
    """Boolean mask of the union of ``segnet.roi_classes`` at ``postprocess_mpp``, shape as ``wsi_processing.mask_shape``."""

    cfg = settings["segnet"]
    classes, mpp = model["classes"], model["mpp"]
    keep = [classes.index(c) for c in cfg["roi_classes"]]
    tile, margin, views = cfg["tile_um"], cfg["margin_um"], cfg["tta_views"]
    w0, h0 = slide.dimensions
    w_um, h_um = math.ceil(w0 * mpp0[0]), math.ceil(h0 * mpp0[1])
    cell = CELL_STRIDE * mpp
    mask = np.zeros((math.ceil(h_um / cell), math.ceil(w_um / cell)), dtype=bool)
    for ty in range(0, h_um, tile):
        for tx in range(0, w_um, tile):
            x0, y0 = max(tx - margin, 0), max(ty - margin, 0)
            x1, y1 = min(tx + tile + margin, w_um), min(ty + tile + margin, h_um)
            img = read_um(slide, mpp0, x0, y0, x1 - x0, y1 - y0, mpp)
            if tissue_fraction(img) < cfg["min_tissue_frac"]:
                continue                                         # glass: no class of interest
            probs = predict_cells(img, model["nets"], model["window"], device, views, cfg["batch_size"])
            label = probs.argmax(0).cpu().numpy()
            cx, cy = int((tx - x0) / cell), int((ty - y0) / cell)
            part = np.isin(label, keep)[cy:cy + int(tile / cell), cx:cx + int(tile / cell)]
            gx, gy = int(tx / cell), int(ty / cell)
            h, w = mask[gy:gy + part.shape[0], gx:gx + part.shape[1]].shape
            mask[gy:gy + h, gx:gx + w] = part[:h, :w]
    target = float(settings["postprocess_mpp"])
    resampled = resample_mask(mask, (math.ceil(h0 * mpp0[1] / target), math.ceil(w0 * mpp0[0] / target)))
    return fill_holes(resampled)
