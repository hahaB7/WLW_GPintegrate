from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from components.person_c import wsi_processing as wsi
from components.person_c.region import figures
from components.person_c.region.segnet import fill_holes, window_starts

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "region_proposal.json"


class SegnetSettingsTests(unittest.TestCase):
    def config(self) -> dict:
        return json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_shipped_config_is_valid(self) -> None:
        settings = wsi.validate_settings(self.config())
        self.assertEqual("region_proposal", settings["mode"])
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
        with self.assertRaises(ValueError):
            wsi.validate_figures({"dir": "x", "annotations": 3})
        self.assertIsNone(wsi.validate_figures({"dir": "x"})["annotations"])

    def test_only_the_switched_stages_are_written_at_the_size_of_the_stages(self) -> None:
        import tempfile

        import cv2

        labels = np.full((3000, 4000), 255, dtype=np.uint8)           # 1 um/px: a 4 mm x 3 mm slide
        labels[500:2500, 500:2000] = 0
        labels[500:2500, 2000:3500] = 2
        stages = {"tissue": labels, "epithelium": labels == 0, "classes": ["TUM", "BEN", "STR"],
                  "image": np.full((3000, 4000, 3), 235, dtype=np.uint8)}
        with tempfile.TemporaryDirectory() as tmp:
            paths = figures.save_figures((1.0, 1.0), 1.0, stages, [[800, 600, 400, 300]],
                                         {"dir": tmp, "tissue": True, "epithelium": False, "boxes": True}, "S1", "cpu")
            names = sorted(Path(p).name for p in paths)
            self.assertEqual(["S1_boxes.jpg", "S1_tissue.jpg"], names)
            self.assertEqual(names, sorted(p.name for p in Path(tmp).iterdir()))
            image = cv2.imread(str(Path(tmp) / "S1_boxes.jpg"))
            self.assertEqual((3000, 4000, 3), image.shape)
            self.assertGreater(int(image[600, 1000, 2]), 200)          # the top edge of the box is red (BGR channel 2)
            self.assertLess(int(image[600, 1000, 0]), 60)
            self.assertEqual([], figures.save_figures((1.0, 1.0), 1.0, stages, [], {"dir": tmp}, "S2", "cpu"))

    def test_figure_options_downscale_quality_thickness_and_validation(self) -> None:
        import tempfile

        import cv2

        labels = np.full((3001, 4003), 255, dtype=np.uint8)           # sizes that are not multiples of the downscale
        labels[500:2500, 500:2000] = 0
        stages = {"tissue": labels, "epithelium": labels == 0, "classes": ["TUM", "BEN"],
                  "image": np.full((3001, 4003, 3), 235, dtype=np.uint8)}
        config = wsi.validate_figures({"dir": "x", "boxes": True, "downscale": 4, "quality": 70, "box_thickness": 3})
        with tempfile.TemporaryDirectory() as tmp:
            config["dir"] = tmp
            figures.save_figures((1.0, 1.0), 1.0, stages, [[800, 600, 400, 300]], config, "S", "cpu")
            image = cv2.imread(str(Path(tmp) / "S_boxes.jpg"))
            self.assertEqual((751, 1001, 3), image.shape)                # 4 um/px
            top = image[150, 200:300]                                    # the top edge of the box at (800, 600) -> (200, 150)
            red = (top[:, 0] < 100) & (top[:, 2] > 200)
            self.assertTrue(red[2:16].all() and not red[21:33].any())      # a dash of 18 px (6 x thickness 3), then a gap of 18
        for bad in ({"downscale": 0}, {"quality": 101}, {"box_thickness": 0}):
            with self.assertRaises(ValueError):
                wsi.validate_figures({"dir": "x", **bad})

    def test_annotation_rois_are_green_and_only_drawn_when_there_are_some(self) -> None:
        import tempfile

        import cv2

        stages = {"tissue": np.full((600, 800), 255, dtype=np.uint8), "epithelium": np.zeros((600, 800), dtype=bool),
                  "classes": ["TUM"], "image": np.full((600, 800, 3), 235, dtype=np.uint8)}
        switches = {"boxes": True}
        with tempfile.TemporaryDirectory() as tmp:
            switches["dir"] = tmp
            figures.save_figures((1.0, 1.0), 1.0, stages, [], switches, "With", "cpu", [[100, 100, 200, 150]])
            figures.save_figures((1.0, 1.0), 1.0, stages, [], switches, "Without", "cpu")
            with_ann = cv2.imread(str(Path(tmp) / "With_boxes.jpg"))
            without = cv2.imread(str(Path(tmp) / "Without_boxes.jpg"))
            pixel = with_ann[100, 200]                                    # the top edge of the annotation (BGR)
            self.assertTrue(pixel[1] > 120 and pixel[0] < 60 and pixel[2] < 60)
            self.assertLess(int(np.abs(without[100, 200].astype(int) - 235).max()), 12)   # nothing there without it


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
    def test_fill_holes_equals_scipy_on_random_masks(self) -> None:
        from scipy import ndimage

        rng = np.random.default_rng(0)
        for density in (0.3, 0.5, 0.6, 0.7):
            mask = rng.random((200, 300)) < density
            self.assertTrue(np.array_equal(fill_holes(mask), ndimage.binary_fill_holes(mask)))

    def test_window_starts_cover_the_axis(self) -> None:
        self.assertEqual([0, 112, 224, 300], window_starts(524, 224, 112))
        self.assertEqual([0], window_starts(100, 224, 112))

    def test_boxes_follow_the_mask_in_level0_pixels(self) -> None:
        settings = {"postprocess_mpp": 1.0, "min_object_size": 100}
        mask = np.zeros((400, 600), dtype=bool)
        mask[100:150, 100:200] = True
        mask[100:150, 201:300] = True                  # one pixel apart: two objects
        mask[300:305, 50:55] = True                    # 25 px: below min_object_size
        boxes = wsi.level0_boxes(mask, (0.5, 0.5), (1200, 800), settings)
        self.assertEqual([[200, 200, 200, 100], [402, 200, 198, 100]], boxes)   # um x 0.5 -> level-0 px: x 2 per um


if __name__ == "__main__":
    unittest.main()
