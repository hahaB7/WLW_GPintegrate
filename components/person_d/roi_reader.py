"""Read E ROI pixels from E's stain filepath and ROI geometry.

``main_info.roi_path`` is used when E provides a crop; it must stay below
sibling ``run/`` and its pixel size must equal ``main_info.roi_wh`` (no
resampling).  Otherwise the WSI at ``stains[].filepath`` is read at
``level0_info.xywh`` and resampled to ``main_info.roi_wh``.  Only an absolute
WSI filepath may point outside ``run/``.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image

from components.person_d.geometry import roi_geometry
from contracts.paths import resolve_run_path


def read_wsi_roi(slide: Any, roi: Mapping[str, Any]) -> Image.Image:
    (x, y, width, height), target_size, desired = roi_geometry(roi)
    if x + width > slide.dimensions[0] or y + height > slide.dimensions[1]:
        raise ValueError(f"ROI {roi['roi_id']} exceeds the WSI dimensions")

    level = slide.get_best_level_for_downsample(max(1.0, desired))
    downsample = float(slide.level_downsamples[level])
    source_size = (math.ceil(width / downsample), math.ceil(height / downsample))
    rgba = slide.read_region((x, y), level, source_size)
    try:
        image = rgba.convert("RGB")
    finally:
        rgba.close()
    if image.size != target_size:
        resized = image.resize(target_size, Image.Resampling.BICUBIC)
        image.close()
        image = resized
    return image


def _open_openslide(path: str) -> Any:
    import openslide

    return openslide.OpenSlide(path)


class ROIImageReader:
    """Open each WSI once per run and return RGB ROI images."""

    def __init__(self, open_slide: Callable[[str], Any] = _open_openslide):
        self._open_slide = open_slide
        self._slides: dict[Path, Any] = {}
        self._stack = ExitStack()

    def __enter__(self) -> "ROIImageReader":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._stack.close()
        self._slides.clear()

    def read(self, stain: Mapping[str, Any], roi: Mapping[str, Any]) -> Image.Image:
        _, target_size, _ = roi_geometry(roi)
        roi_path = roi["main_info"].get("roi_path")
        if roi_path:
            path = resolve_run_path(roi_path)
            if not path.is_file():
                raise FileNotFoundError(f"ROI image is missing: {path}")
            with Image.open(path) as source:
                if source.size != target_size:
                    raise ValueError(
                        f"ROI {roi['roi_id']} crop {path} is {source.size[0]} x {source.size[1]} px, "
                        f"but main_info.roi_wh is {target_size[0]} x {target_size[1]}"
                    )
                return source.convert("RGB")

        raw = Path(str(stain["filepath"])).expanduser()
        wsi_path = raw.resolve() if raw.is_absolute() else resolve_run_path(raw)
        if not wsi_path.exists():
            raise FileNotFoundError(
                f"WSI is missing for stain {stain['stain_id']}: {wsi_path}"
            )
        if wsi_path not in self._slides:
            slide = self._open_slide(str(wsi_path))
            if hasattr(slide, "close"):
                self._stack.callback(slide.close)
            self._slides[wsi_path] = slide
        return read_wsi_roi(self._slides[wsi_path], roi)
