from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from components.person_c import wsi_processing as wsi
from components.person_c.region import figures
from components.person_c.region.segnet import fill_holes, resample_mask, window_starts

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.json"


class SegnetSettingsTests(unittest.TestCase):
    def config(self) -> dict:
        return json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_shipped_config_is_valid(self) -> None:
        settings = wsi.validate_settings(self.config())
        self.assertEqual("segnet", settings["backend"])
        self.assertEqual(["TUM", "BEN"], settings["segnet"]["roi_classes"])

    def test_bad_values_are_rejected(self) -> None:
        for path, value in (
            (("segnet", "tile_um"), 1790),
            (("segnet", "tta_views"), 9),
            (("segnet", "checkpoints"), [{"path": "x.pt", "revision": "abc"}]),
            (("segnet", "roi_classes"), []),
        ):
            config = self.config()
            config[path[0]][path[1]] = value
            with self.assertRaises(ValueError, msg=str(path)):
                wsi.validate_settings(config)


class ReadTests(unittest.TestCase):
    def test_a_level_one_percent_coarser_is_still_used(self) -> None:
        from PIL import Image

        from components.person_c.region.segnet import read_um

        class Slide:
            level_downsamples = [1.0, 4.0, 16.0]
            levels_read: list[int] = []

            def read_region(self, origin, level, size):
                self.levels_read.append(level)
                return Image.new("RGBA", size, (200, 100, 150, 255))

        slide = Slide()
        img = read_um(slide, (0.2524, 0.2524), 100, 100, 200, 120, 1.0)   # 3.96 px per um: the level of 4.0 is the one
        self.assertEqual([1], slide.levels_read)
        self.assertEqual((120, 200, 3), img.shape)


class FigureTests(unittest.TestCase):
    def test_switches_are_validated(self) -> None:
        self.assertFalse(wsi.validate_figures(None)["boxes"])
        with self.assertRaises(ValueError):
            wsi.validate_figures({"dir": "x", "tissue": "yes"})
        with self.assertRaises(ValueError):
            wsi.validate_figures({"boxes": True})                  # a switch on needs a directory

    def test_only_the_switched_stages_are_written(self) -> None:
        import tempfile

        class Slide:
            dimensions = (4000, 3000)

            def get_thumbnail(self, size):
                from PIL import Image
                return Image.new("RGB", (1000, 750), (240, 200, 220))

        labels = np.full((30, 40), 255, dtype=np.uint8)
        labels[5:25, 5:20] = 0
        labels[5:25, 20:35] = 2
        stages = {"tissue": labels, "epithelium": labels == 0, "classes": ["TUM", "BEN", "STR"]}
        with tempfile.TemporaryDirectory() as tmp:
            paths = figures.save_figures(Slide(), stages, [[800, 600, 400, 300]],
                                         {"dir": tmp, "tissue": True, "epithelium": False, "boxes": True}, "c1", "s1")
            names = sorted(Path(p).name for p in paths)
            self.assertEqual(["c1_s1_boxes.png", "c1_s1_tissue.png"], names)
            self.assertEqual(names, sorted(p.name for p in Path(tmp).iterdir()))
            self.assertEqual([], figures.save_figures(Slide(), stages, [], {"dir": tmp}, "c1", "s2"))


class EnclosedTests(unittest.TestCase):
    def test_only_a_box_completely_inside_another_is_dropped(self) -> None:
        big = [0, 0, 1000, 1000]
        boxes = [[100, 100, 50, 50], big, [950, 100, 100, 100], [0, 0, 1000, 1000], [2000, 2000, 10, 10], [0, 500, 1000, 500]]
        # [950,100,100,100] sticks out of big by 50 px, so it stays; the copy of big and the box touching its edge go
        self.assertEqual([big, [950, 100, 100, 100], [2000, 2000, 10, 10]], wsi.drop_enclosed(boxes))


class HoleTests(unittest.TestCase):
    def test_a_ring_is_filled_with_what_is_inside_and_a_border_gap_is_not(self) -> None:
        mask = np.zeros((60, 60), dtype=bool)
        mask[5:50, 5:50] = True
        mask[15:40, 15:40] = False          # the hole of a ring ...
        mask[25:30, 25:30] = True           # ... with another object in it
        mask[5:50, 54:58] = True            # a separate object
        out = fill_holes(mask)
        self.assertTrue(out[5:50, 5:50].all())
        self.assertTrue(out[5:50, 54:58].all() and not out[:, 50:54].any())


class MaskTests(unittest.TestCase):
    def test_resample_keeps_the_region_and_the_requested_shape(self) -> None:
        cells = np.zeros((100, 100), dtype=bool)      # 4 um cells
        cells[20:60, 10:50] = True
        out = resample_mask(cells, (50, 50))          # 8 um pixels
        self.assertEqual((50, 50), out.shape)
        self.assertTrue(out[10:30, 5:25].all())
        self.assertFalse(out[:9].any() or out[31:].any())

    def test_resample_drops_a_half_covered_pixel_edge_case(self) -> None:
        cells = np.zeros((4, 4), dtype=bool)
        cells[0, 0] = True                            # one cell of four of its pixel
        self.assertFalse(resample_mask(cells, (2, 2)).any())

    def test_window_starts_cover_the_axis(self) -> None:
        self.assertEqual([0, 112, 224, 300], window_starts(524, 224, 112))
        self.assertEqual([0], window_starts(100, 224, 112))


if __name__ == "__main__":
    unittest.main()
