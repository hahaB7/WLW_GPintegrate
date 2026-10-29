from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from components.person_e.embedding_catalog import (
    embedding_filename,
    load_embedding_bank,
)
from components.person_e.native_inference import (
    _pseudo_prediction,
    _read_wsi_roi,
    _resolve_data_path,
    resolve_native_artifacts,
)
from contracts.paths import REFERENCE_ROOT, RUN_ROOT


class _FakeSlide:
    dimensions = (2000, 2000)
    level_downsamples = (1.0, 2.0, 4.0)

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return 1 if downsample >= 2 else 0

    def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]):
        return Image.new("RGBA", size, (120, 80, 60, 255))


class NativeInferenceTests(unittest.TestCase):
    def test_manifest_catalog_reassembles_per_result_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filename = embedding_filename("pretrain", "D", "A")
            (root / "embedding_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "clee_dxitem_embedding_catalog_v1"
                        ),
                        "dx_item_order": ["D"],
                        "dx_result_order": {"D": ["A"]},
                        "entries": [
                            {
                                "type": "pretrain",
                                "dx_item": "D",
                                "dx_result": "A",
                                "filename": filename,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            torch.save(
                {
                    "type": "pretrain",
                    "content": {
                        "is": {
                            "embedding": {
                                "medgemma-1.5-4b-it": {
                                    "eosToken": torch.tensor([1.0, 2.0])
                                }
                            }
                        }
                    },
                },
                root / filename,
            )

            bank = load_embedding_bank(
                root,
                ["is", "eosToken"],
                "medgemma-1.5-4b-it",
            )

        self.assertTrue(torch.equal(bank["D"]["A"], torch.tensor([1.0, 2.0])))

    def test_only_absolute_wsi_paths_may_escape_run(self) -> None:
        external = Path("/external/slide.svs")
        self.assertEqual(
            external,
            _resolve_data_path(
                str(external),
                None,
                "stain.filepath",
                allow_external_absolute=True,
            ),
        )
        with self.assertRaisesRegex(ValueError, "Run path must stay"):
            _resolve_data_path("/external/roi.png", None, "main_info.roi_path")
        self.assertEqual(
            RUN_ROOT / "output/roi.png",
            _resolve_data_path(
                "roi.png", RUN_ROOT / "output", "main_info.roi_path"
            ),
        )

    def test_case_ceiling_and_importance_are_independent(self) -> None:
        labels = ["Normal", "Benign", "Atypia", "Cancer"]
        prediction = _pseudo_prediction(
            scores={"Normal": 0.8, "Benign": 0.7, "Atypia": 0.65, "Cancer": 0.9},
            labels=labels,
            thresholds={label: 0.5 for label in labels},
            case_label="Atypia",
            case_threshold=0.6,
            chunk_idx=2,
        )
        self.assertEqual("Normal", prediction["assigned"])
        self.assertTrue(prediction["assigned_as_ref"])
        self.assertEqual(0.65, prediction["importance"])
        self.assertEqual(2, prediction["chunk_idx"])

    def test_wsi_crop_resamples_level_zero_rectangle(self) -> None:
        image = _read_wsi_roi(
            _FakeSlide(),
            {
                "roi_id": "roi-1",
                "level0_info": {"xywh": [0, 0, 1000, 800], "mpp": [0.25, 0.25]},
                "main_info": {"mpp": 0.5},
            },
        )
        self.addCleanup(image.close)
        self.assertEqual((500, 400), image.size)
        self.assertEqual("RGB", image.mode)

    def test_checkpoint_manifest_selects_matching_threshold_bundle(self) -> None:
        with tempfile.TemporaryDirectory(
            dir=REFERENCE_ROOT / "person_e/checkpoint"
        ) as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (root / "prefix.pth").write_bytes(b"test")
            (root / "[checkpoint]best_model.json").write_text(
                json.dumps(
                    {
                        "checkpoint_metadata": {"epoch_idx": 7},
                        "model_filename": "prefix.pth",
                    }
                ),
                encoding="utf-8",
            )
            bundle = root / "threshold_bundle[007]_fixture.json"
            bundle.write_text(json.dumps({"epoch_idx": 7}), encoding="utf-8")
            artifacts = resolve_native_artifacts({"checkpoint_dir": str(root)})
            self.assertEqual(bundle, artifacts.threshold_bundle_path)
            self.assertEqual(7, artifacts.checkpoint_epoch)


if __name__ == "__main__":
    unittest.main()
