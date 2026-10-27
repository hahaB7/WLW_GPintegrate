from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from components.person_d.roi_reader import ROIImageReader, read_wsi_roi, roi_geometry
from contracts.paths import RUN_ROOT


class _FakeSlide:
    dimensions = (4000, 4000)
    level_downsamples = (1.0, 2.0, 4.0)

    def __init__(self) -> None:
        self.closed = False
        self.reads: list[tuple[tuple[int, int], int, tuple[int, int]]] = []

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return 1 if downsample >= 2 else 0

    def read_region(self, location, level, size):
        self.reads.append((location, level, size))
        return Image.new("RGBA", size, (120, 80, 60, 255))

    def close(self) -> None:
        self.closed = True


def _roi(xywh=(100, 200, 1000, 800), level0_mpp=0.25, mpp=0.5, roi_wh=(500, 400), roi_path=None):
    return {
        "roi_id": "roi-1",
        "level0_info": {"xywh": list(xywh), "mpp": [level0_mpp, level0_mpp]},
        "main_info": {"mpp": mpp, "roi_wh": list(roi_wh), "roi_path": roi_path},
    }


class ROIReaderTests(unittest.TestCase):
    def test_wsi_crop_reads_level_zero_rectangle_at_main_size(self) -> None:
        slide = _FakeSlide()
        image = read_wsi_roi(slide, _roi())
        self.addCleanup(image.close)
        self.assertEqual((500, 400), image.size)
        self.assertEqual("RGB", image.mode)
        self.assertEqual([((100, 200), 1, (500, 400))], slide.reads)

    def test_inconsistent_or_invalid_geometry_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "roi_wh"):
            roi_geometry(_roi(roi_wh=(256, 256)))
        with self.assertRaisesRegex(ValueError, "xywh"):
            roi_geometry(_roi(xywh=(0, 0, 0, 800)))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            read_wsi_roi(_FakeSlide(), _roi(xywh=(3500, 0, 1000, 800)))

    def test_each_wsi_opens_once_and_closes_on_exit(self) -> None:
        slides: dict[str, _FakeSlide] = {}

        def open_slide(path: str) -> _FakeSlide:
            slides[path] = _FakeSlide()
            return slides[path]

        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as handle:
            wsi = Path(handle.name)
        self.addCleanup(wsi.unlink)
        stain = {"stain_id": "stain-1", "filepath": str(wsi)}
        with ROIImageReader(open_slide) as reader:
            reader.read(stain, _roi()).close()
            reader.read(stain, _roi(xywh=(0, 0, 1000, 800))).close()
        self.assertEqual(1, len(slides))
        self.assertTrue(next(iter(slides.values())).closed)

    def test_missing_wsi_and_paths_outside_run_fail(self) -> None:
        reader = ROIImageReader(lambda path: _FakeSlide())
        with self.assertRaises(FileNotFoundError):
            reader.read({"stain_id": "s", "filepath": str(RUN_ROOT.parent / "missing.svs")}, _roi())
        with self.assertRaisesRegex(ValueError, "Run path must stay"):
            reader.read({"stain_id": "s", "filepath": "relative/outside.svs"}, _roi())
        with self.assertRaisesRegex(ValueError, "Run path must stay"):
            reader.read({"stain_id": "s", "filepath": "unused"}, _roi(roi_path="crops/roi.png"))

    def test_roi_crop_below_run_is_used(self) -> None:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RUN_ROOT) as directory:
            crop = Path(directory) / "roi.png"
            Image.new("RGB", (500, 400), (10, 20, 30)).save(crop)
            reader = ROIImageReader(lambda path: self.fail("WSI must not be opened"))
            image = reader.read({"stain_id": "s", "filepath": "unused"}, _roi(roi_path=str(crop)))
            self.addCleanup(image.close)
            self.assertEqual((10, 20, 30), image.getpixel((0, 0)))

    def test_roi_crop_with_another_size_than_roi_wh_fails(self) -> None:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RUN_ROOT) as directory:
            crop = Path(directory) / "roi.png"
            Image.new("RGB", (13, 17)).save(crop)
            reader = ROIImageReader(lambda path: self.fail("WSI must not be opened"))
            with self.assertRaisesRegex(ValueError, "13 x 17 px, but main_info.roi_wh is 500 x 400"):
                reader.read({"stain_id": "s", "filepath": "unused"}, _roi(roi_path=str(crop)))


if __name__ == "__main__":
    unittest.main()
