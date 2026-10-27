from __future__ import annotations

import json
import unittest
from pathlib import Path

from components.person_c import assets
from components.person_c.interest_pattern import build_region_artifact
from components.person_c.rois import build_roi_info, plan_boxes
from contracts.runtime import ContractError, validate_artifact, validate_case_list_input

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FIXTURE = EXAMPLES / "cases.example.json"
SLIDE = (10000, 8000)


def _poly(x0, y0, x1, y1, holes=()):
    ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    return {"geometry": {"type": "Polygon", "coordinates": [ring, *holes]}}


class PlanBoxesTests(unittest.TestCase):
    def test_box_is_the_component_bounding_box(self) -> None:
        boxes = plan_boxes([_poly(1000.4, 1000.6, 1600.2, 1400)], SLIDE, 0, 10)
        self.assertEqual([(1000, 1000, 601, 400)], boxes)

    def test_sizes_vary_and_holes_do_not_change_the_box(self) -> None:
        hole = [[2000, 2000], [3000, 2000], [3000, 3000], [2000, 3000], [2000, 2000]]
        boxes = plan_boxes(
            [_poly(0, 0, 3072, 3072, [hole]), _poly(5000, 4000, 5400, 4300)], SLIDE, 0, 10
        )
        self.assertEqual([(0, 0, 3072, 3072), (5000, 4000, 400, 300)], boxes)

    def test_area_subtracts_holes_for_the_min_filter(self) -> None:
        hole = [[10, 10], [90, 10], [90, 90], [10, 90], [10, 10]]
        self.assertEqual([], plan_boxes([_poly(0, 0, 100, 100, [hole])], SLIDE, 5000, 10))
        self.assertEqual(1, len(plan_boxes([_poly(0, 0, 100, 100, [hole])], SLIDE, 3000, 10)))

    def test_tiny_component_is_dropped(self) -> None:
        self.assertEqual([], plan_boxes([_poly(0, 0, 100, 100)], SLIDE, 20000, 10))

    def test_clamped_to_slide_sorted_and_largest_kept_when_capped(self) -> None:
        features = [
            _poly(9900, 7900, 10200, 8300),   # sticks out of the slide
            _poly(0, 5000, 100, 5100),
            _poly(2000, 100, 4000, 2000),
            _poly(100, 100, 150, 150),
        ]
        boxes = plan_boxes(features, SLIDE, 0, 3)
        self.assertEqual([(2000, 100, 2000, 1900), (0, 5000, 100, 100),
                          (9900, 7900, 100, 100)], boxes)
        for left, top, width, height in boxes:
            self.assertTrue(0 <= left and left + width <= SLIDE[0])
            self.assertTrue(0 <= top and top + height <= SLIDE[1])


class RoiInfoTests(unittest.TestCase):
    def test_main_info_follows_person_e_resampling_rule(self) -> None:
        level0, main = build_roi_info((128, 128, 1024, 1024), (0.25, 0.25), 0.5)
        self.assertEqual(0.25, level0["mpp"])
        self.assertEqual([128, 128, 1024, 1024], level0["xywh"])
        self.assertEqual([512, 512], main["roi_wh"])
        self.assertEqual([64, 64, 512, 512], main["xywh"])
        self.assertEqual(512 * 512, main["area"])

    def test_large_roi_is_viewed_coarser_but_level0_is_untouched(self) -> None:
        level0, main = build_roi_info((0, 0, 25000, 20000), (0.25, 0.25), 0.5, 2048)
        self.assertEqual([25000, 20000], level0["roi_wh"])
        self.assertEqual(25000 * 0.25 / 2048, main["mpp"])
        self.assertEqual(2048, max(main["roi_wh"]))
        small0, small = build_roi_info((0, 0, 1000, 1000), (0.25, 0.25), 0.5, 2048)
        self.assertEqual(0.5, small["mpp"])
        self.assertEqual([500, 500], small["roi_wh"])

    def test_anisotropic_mpp_is_a_pair(self) -> None:
        level0, main = build_roi_info((0, 0, 1000, 1000), (0.25, 0.5), 0.5)
        self.assertEqual([0.25, 0.5], level0["mpp"])
        self.assertEqual([500, 1000], main["roi_wh"])


class RegionArtifactTests(unittest.TestCase):
    config = {
        "component_version": "person-c/test",
        "mode": "region_proposal",
        "stain_types": ["EXAMPLE"],
        "region": {},
        "roi": {
            "target_mpp": 0.5,
            "min_region_area_px": 0,
            "max_main_side_px": 2048,
            "max_rois_per_stain": 10,
        },
    }

    def _source(self) -> dict:
        return json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_artifact_validates_and_ids_are_unique(self) -> None:
        features = [_poly(100, 100, 600, 600), _poly(5000, 4000, 5400, 4300)]
        artifact = build_region_artifact(
            self._source(),
            self.config,
            propose=lambda path, region: features,
            slide_info=lambda path: (SLIDE, (0.25, 0.25)),
        )
        validate_artifact(artifact, "E.ROIs")
        stain = artifact["payload"]["case_list"][0]["tissue_blocks"][0]["stains"][0]
        self.assertEqual(2, stain["roi_num"])
        self.assertEqual(["example-stain-roi-00000", "example-stain-roi-00001"],
                         [r["roi_id"] for r in stain["roi_list"]])
        self.assertEqual([0, 1], [r["global_idx"] for r in stain["roi_list"]])

    def test_unprocessed_stain_type_keeps_empty_roi_list(self) -> None:
        artifact = build_region_artifact(
            self._source(),
            {**self.config, "stain_types": ["HE"]},
            propose=lambda path, region: self.fail("must not run"),
            slide_info=lambda path: self.fail("must not open"),
        )
        validate_artifact(artifact, "E.ROIs")
        stain = artifact["payload"]["case_list"][0]["tissue_blocks"][0]["stains"][0]
        self.assertEqual((0, []), (stain["roi_num"], stain["roi_list"]))

    def test_slide_failure_propagates(self) -> None:
        def broken(path):
            raise ValueError("no MPP")

        with self.assertRaisesRegex(ValueError, "no MPP"):
            build_region_artifact(
                self._source(), self.config, propose=lambda p, r: [], slide_info=broken
            )


EXAMPLE_SLIDE = ((30000, 20000), (0.25, 0.25))
EXAMPLE_CONFIG = {
    "component_version": "person-c/example",
    "mode": "region_proposal",
    "stain_types": ["HE"],
    "region": {"vocab": "BCSS"},
    "roi": {
        "target_mpp": 0.5,
        "min_region_area_px": 262144,
        "max_rois_per_stain": 256,
        "max_main_side_px": 2048,
    },
}


class CanonicalExampleTests(unittest.TestCase):
    def test_example_matches_expected_output(self) -> None:
        source = json.loads((EXAMPLES / "case_list.valid.json").read_text(encoding="utf-8"))
        features = json.loads((EXAMPLES / "proposals.valid.json").read_text(encoding="utf-8"))
        artifact = build_region_artifact(
            source,
            EXAMPLE_CONFIG,
            propose=lambda path, region: features["features"],
            slide_info=lambda path: EXAMPLE_SLIDE,
        )
        validate_artifact(artifact, "E.ROIs")
        expected = json.loads((EXAMPLES / "E_rois.expected.json").read_text(encoding="utf-8"))
        self.assertEqual(expected, artifact)

    def test_example_covers_the_documented_cases(self) -> None:
        expected = json.loads((EXAMPLES / "E_rois.expected.json").read_text(encoding="utf-8"))
        stains = expected["payload"]["case_list"][0]["tissue_blocks"][0]["stains"]
        he, ihc = stains
        self.assertEqual((4, []), (he["roi_num"], ihc["roi_list"]))
        self.assertEqual(0, ihc["roi_num"])
        main_sides = [max(r["main_info"]["roi_wh"]) for r in he["roi_list"]]
        self.assertEqual(2048, max(main_sides))
        for roi in he["roi_list"]:
            left, top, width, height = roi["level0_info"]["xywh"]
            self.assertTrue(left + width <= 30000 and top + height <= 20000)


class InvalidInputTests(unittest.TestCase):
    def test_multi_case_input_is_rejected(self) -> None:
        source = json.loads(FIXTURE.read_text(encoding="utf-8"))
        second = json.loads(json.dumps(source["case_list"][0]))
        second["case_id"] = "example-case-2"
        source["case_list"].append(second)
        with self.assertRaisesRegex(ContractError, "exactly one case"):
            validate_case_list_input(source, single_case=True)

    def test_missing_slide_file_fails(self) -> None:
        try:
            import openslide  # noqa: F401
        except ImportError:
            self.skipTest("openslide is not installed in this environment")
        from components.person_c.interest_pattern import slide_info

        with self.assertRaisesRegex(FileNotFoundError, "WSI is missing"):
            slide_info("/nonexistent/slide.svs")


class AssetVerificationTests(unittest.TestCase):
    def _region(self, root: Path) -> dict:
        lib = root / "CONCH"
        lib.mkdir()
        (lib / "a.py").write_text("x", encoding="utf-8")
        checkpoint = root / "pytorch_model.bin"
        checkpoint.write_bytes(b"weights")
        return {"conch_lib_path": str(lib), "conch_checkpoint": str(checkpoint),
                "vocab_json": str(Path(assets.__file__).parent / "region/vocab/bcss_vocab.json")}

    def _pin(self, region: dict):
        from unittest import mock

        return mock.patch.multiple(
            assets,
            CONCH_LIBRARY_TREE_SHA256=assets.tree_sha256(Path(region["conch_lib_path"])),
            CONCH_CHECKPOINT_SHA256=assets.file_sha256(Path(region["conch_checkpoint"])),
        )

    def test_shipped_vocabulary_matches_its_pin(self) -> None:
        vocab = Path(assets.__file__).parent / "region/vocab/bcss_vocab.json"
        self.assertEqual(assets.BCSS_VOCAB_SHA256, assets.file_sha256(vocab))

    def test_shipped_conch_library_matches_its_pin(self) -> None:
        lib = Path(assets.__file__).parent / "third_party/CONCH"
        self.assertEqual(assets.CONCH_LIBRARY_TREE_SHA256, assets.tree_sha256(lib))

    def test_matching_assets_pass(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            region = self._region(Path(tmp))
            with self._pin(region):
                assets.verify_assets(region)

    def test_changed_checkpoint_fails(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            region = self._region(Path(tmp))
            with self._pin(region):
                Path(region["conch_checkpoint"]).write_bytes(b"other")
                with self.assertRaisesRegex(assets.AssetError, "CONCH checkpoint digest"):
                    assets.verify_assets(region)

    def test_changed_library_fails(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            region = self._region(Path(tmp))
            with self._pin(region):
                (Path(region["conch_lib_path"]) / "a.py").write_text("y", encoding="utf-8")
                with self.assertRaisesRegex(assets.AssetError, "CONCH library digest"):
                    assets.verify_assets(region)

    def test_missing_checkpoint_fails(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            region = self._region(Path(tmp))
            Path(region["conch_checkpoint"]).unlink()
            with self.assertRaisesRegex(assets.AssetError, "checkpoint is missing"):
                assets.verify_assets(region)


if __name__ == "__main__":
    unittest.main()
