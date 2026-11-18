from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from contracts.paths import RUN_ROOT
from contracts.runtime import validate_artifact, validate_case_list_input
from components.person_c.interest_pattern import run
from components.person_c import wsi_processing as wsi


WORK = RUN_ROOT / "output" / "work"
WORK.mkdir(parents=True, exist_ok=True)


def _case_input(he_path: str) -> dict:
    return {
        "DxItem_list": ["Histologic_Type"], "case_list": [{
                "sample_idx": 0, "case_id": "test-case", "hospital": "synthetic",
                "patient_info": {}, "date": "", "source": {}, "organ": "breast",
                "report_raw_content": "Pathologic diagnosis: Unremarkable.",
                "tissue_blocks": [{"block_id": "A", "memo": "", "stains": [
                    {"stain_id": "he-1", "stain_type": "HE", "filename": "tiny.svs",
                     "filepath": he_path, "memo": ""},
                    {"stain_id": "er-1", "stain_type": "ER", "filename": "missing-er.svs",
                     "filepath": "/missing-er.svs", "memo": ""},
                ]}], "memo": "",
            }],
    }


def _config(save_png: bool = False, output_dir: str = "") -> dict:
    return {
        "component_version": "person-c/interest-pattern:test", "mode": "region_proposal",
        "stain_type_whitelist": ["HE"],
        "segnet": {
            "checkpoint": {"path": "../reference/person_c/checkpoint/segnet/missing-test.pt", "revision": "sha256:" + "0" * 64},
            "roi_classes": ["TUM", "BEN"], "tile_um": 1792, "margin_um": 112, "batch_size": 1, "num_workers": 0,
        },
        "device": "cpu", "postprocess_mpp": 8,
        "min_object_size": 1, "save_roi_png": save_png,
        "roi_output_dir": output_dir,
    }


def _stages(mask: np.ndarray) -> dict:
    return {"tissue": np.full(mask.shape, 255, dtype=np.uint8), "epithelium": mask, "classes": ["TUM", "BEN"]}


class _Slide:
    dimensions = (128, 128)
    level_dimensions = ((128, 128),)
    level_downsamples = (1.0,)
    properties = {"openslide.mpp-x": "1", "openslide.mpp-y": "1"}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None


class InterestPatternTest(TestCase):
    def test_he_roi_geometry_and_non_he_preservation(self) -> None:
        with TemporaryDirectory(dir=WORK) as temp_dir:
            he_path = Path(temp_dir) / "tiny.svs"
            he_path.touch()
            original = _case_input(str(he_path))
            validate_case_list_input(original, single_case=True)
            mask = np.zeros((16, 16), dtype=bool)
            mask[1:4, 2:6] = True
            with patch("components.person_c.region.segnet.load_model", return_value=(object(), "cpu")), \
                 patch("openslide.OpenSlide", return_value=_Slide()), \
                 patch("components.person_c.region.segnet.infer_stages", return_value=_stages(mask)):
                output = run(original, _config())
                repeated = run(original, _config())
            validate_artifact(output, "E.ROIs")
            used = output["payload"]["reference_versions"]["interest_pattern"]      # what produced the ROIs
            self.assertEqual(["TUM", "BEN"], used["segnet"]["roi_classes"])
            self.assertTrue(used["segnet"]["checkpoint"]["revision"].startswith("sha256:"))
            self.assertEqual(output, repeated)
            self.assertEqual(original, _case_input(str(he_path)))
            he, er = output["payload"]["case_list"][0]["tissue_blocks"][0]["stains"]
            self.assertEqual(1, he["roi_num"])
            self.assertEqual([], er["roi_list"])
            self.assertEqual("/missing-er.svs", er["filepath"])
            roi = he["roi_list"][0]
            self.assertEqual([16, 8, 32, 24], roi["level0_info"]["xywh"])
            self.assertEqual([1.0, 1.0], roi["level0_info"]["mpp"])
            self.assertTrue(all(0 <= value <= 1 for value in roi["level0_info"]["cxcywh"]))
            self.assertIsNone(roi["main_info"]["roi_path"])
            self.assertIsNone(roi["DxPair"])
            self.assertEqual("interest_pattern_extraction", roi["selection_history"][0]["stage"])

    def test_empty_result_and_missing_wsi(self) -> None:
        with TemporaryDirectory(dir=WORK) as temp_dir:
            he_path = Path(temp_dir) / "tiny.svs"
            he_path.touch()
            empty = np.zeros((16, 16), dtype=bool)
            with patch("components.person_c.region.segnet.load_model", return_value=(object(), "cpu")), \
                 patch("openslide.OpenSlide", return_value=_Slide()), \
                 patch("components.person_c.region.segnet.infer_stages", return_value=_stages(empty)):
                output = run(_case_input(str(he_path)), _config())
            validate_artifact(output, "E.ROIs")
            he = output["payload"]["case_list"][0]["tissue_blocks"][0]["stains"][0]
            self.assertEqual((0, []), (he["roi_num"], he["roi_list"]))
            with patch("components.person_c.region.segnet.load_model", return_value=(object(), "cpu")):
                with self.assertRaisesRegex(FileNotFoundError, "WSI is missing"):
                    run(_case_input(str(Path(temp_dir) / "absent.svs")), _config())

    def test_coordinate_semantics_and_png_switch(self) -> None:
        self.assertEqual([2, 10, 8, 20], wsi.main_box([10, 20, 30, 40], (0.25, 0.5), 1.0, (100, 100))[0])
        self.assertEqual([0.25, 0.4, 0.3, 0.4], wsi.normalized_box([10, 20, 30, 40], (100, 100)))
        with TemporaryDirectory(dir=WORK) as temp_dir:
            he_path = Path(temp_dir) / "tiny.svs"
            he_path.touch()
            mask = np.zeros((16, 16), dtype=bool)
            mask[1:4, 2:6] = True
            with patch("components.person_c.region.segnet.load_model", return_value=(object(), "cpu")), \
                 patch("openslide.OpenSlide", return_value=_Slide()), \
                 patch("components.person_c.region.segnet.infer_stages", return_value=_stages(mask)), \
                 patch("components.person_c.wsi_processing.save_png", return_value="/cache/roi.png") as save:
                output = run(_case_input(str(he_path)), _config(True, temp_dir))
            roi = output["payload"]["case_list"][0]["tissue_blocks"][0]["stains"][0]["roi_list"][0]
            self.assertEqual("/cache/roi.png", roi["main_info"]["roi_path"])
            save.assert_called_once()
            validate_artifact(output, "E.ROIs")

    def test_invalid_contract_and_missing_checkpoint_fail(self) -> None:
        root = Path(__file__).resolve().parents[3]
        with TemporaryDirectory(dir=WORK) as temp_dir:
            input_path = Path(temp_dir) / "invalid.json"
            output_path = Path(temp_dir) / "E.json"
            invalid = _case_input("/data/missing.svs")
            invalid["case_list"][0].pop("case_id")
            input_path.write_text(json.dumps(invalid))
            command = [sys.executable, "-m", "components.person_c.interest_pattern",
                       "--input", str(input_path), "--output", str(output_path),
                       "--config", str(root / "components/person_c/configs/example.json")]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("missing required property 'case_id'", result.stderr)
            self.assertFalse(output_path.exists())

            he_path = Path(temp_dir) / "tiny.svs"
            he_path.touch()
            with self.assertRaisesRegex(FileNotFoundError, "SEGNET checkpoint is missing"):
                run(_case_input(str(he_path)), _config())

    def test_example_mode_opens_no_slide_and_makes_no_roi(self) -> None:
        output = run(_case_input("/data/synthetic.svs"), {"component_version": "person-c:test", "mode": "example"})
        validate_artifact(output, "E.ROIs")
        stains = output["payload"]["case_list"][0]["tissue_blocks"][0]["stains"]
        self.assertEqual([0, 0], [stain["roi_num"] for stain in stains])

    def test_stain_whitelist_is_validated(self) -> None:
        config = {"component_version": "person-c:test", "mode": "example", "stain_type_whitelist": []}
        with self.assertRaisesRegex(ValueError, "non-empty string list"):
            run(_case_input("/data/synthetic.svs"), config)


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def canonical_output() -> dict:
    """E of examples/case_list.valid.json in mode region_proposal with a slide and an epithelium mask that are made up here
    (the student is not run): a ring with a small object in its hole (that box is enclosed and dropped), an object at the
    border of the slide, and one object below ``min_object_size`` (no box)."""

    config = json.loads((EXAMPLES.parents[0] / "configs" / "region_proposal.json").read_text(encoding="utf-8"))
    source = json.loads((EXAMPLES / "case_list.valid.json").read_text(encoding="utf-8"))
    mask = np.zeros((7500, 10000), dtype=bool)                  # 1 um/px: a 10 mm x 7.5 mm slide at 0.25 um/px
    mask[1000:1600, 2000:3000] = True
    mask[1100:1500, 2100:2900] = False                          # the ring ...
    mask[1200:1300, 2300:2400] = True                           # ... and an object in its hole
    mask[7400:7500, 9900:10000] = True                          # at the border
    mask[3000:3050, 3000:3050] = True                           # 2500 px: too small
    stages = {"tissue": np.full(mask.shape, 255, dtype=np.uint8), "epithelium": mask, "classes": ["TUM", "BEN"]}

    class Slide(_Slide):
        dimensions = (40000, 30000)
        level_dimensions = ((40000, 30000),)
        properties = {"openslide.mpp-x": "0.25", "openslide.mpp-y": "0.25"}

    with patch("pathlib.Path.is_file", return_value=True), \
         patch("components.person_c.region.segnet.load_model", return_value=(object(), "cpu")), \
         patch("openslide.OpenSlide", return_value=Slide()), \
         patch("components.person_c.region.segnet.infer_stages", return_value=stages):
        return run(source, config)


class CanonicalExampleTests(TestCase):
    def test_expected_e_is_what_the_code_writes(self) -> None:
        output = canonical_output()
        validate_artifact(output, "E.ROIs")
        stains = output["payload"]["case_list"][0]["tissue_blocks"][0]["stains"]
        self.assertEqual([2, 0], [stain["roi_num"] for stain in stains])      # the HE stain; the IHC stain is not processed
        self.assertEqual(json.loads((EXAMPLES / "E_rois.expected.json").read_text(encoding="utf-8")), output)
