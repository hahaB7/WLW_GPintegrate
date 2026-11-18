from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from components.person_c import wsi_processing as wsi
from components.person_c.region.segnet import fill_holes, resample_mask, window_starts

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "segnet.json"


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
            (("segnet", "min_tissue_frac"), 2),
            (("segnet", "checkpoints"), [{"path": "x.pt", "revision": "abc"}]),
            (("segnet", "roi_classes"), []),
        ):
            config = self.config()
            config[path[0]][path[1]] = value
            with self.assertRaises(ValueError, msg=str(path)):
                wsi.validate_settings(config)


class NestedTests(unittest.TestCase):
    def test_inner_and_mostly_inner_boxes_are_dropped(self) -> None:
        big = [0, 0, 1000, 1000]
        boxes = [[100, 100, 50, 50], big, [950, 100, 100, 100], [2000, 2000, 10, 10]]
        self.assertEqual([big, [950, 100, 100, 100], [2000, 2000, 10, 10]], wsi.drop_nested(boxes, 0.9))
        self.assertEqual([big, [2000, 2000, 10, 10]], wsi.drop_nested(boxes, 0.5))


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
