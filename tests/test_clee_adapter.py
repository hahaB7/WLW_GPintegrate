from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from components.person_e.adapter import (
    apply_effective_selection,
    load_checkpoint_support,
    run_fixture_backend,
)
from components.person_e.clee import _merge_pseudo_dx_pair, _prediction_event
from contracts.paths import REFERENCE_ROOT


ROOT = Path(__file__).resolve().parents[1]


class CLEEAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (ROOT / "components/person_e/configs/default.json").read_text(
                encoding="utf-8"
            )
        )
        self.support = load_checkpoint_support(self.config)

    def test_effective_support_is_wlw_policy_intersect_checkpoint(self) -> None:
        self.assertEqual(
            ("Histologic_Type", "Microcalcification"),
            self.support.active_DxItem_list,
        )
        self.assertEqual(
            ("Histologic_Type", "Microcalcification"),
            self.support.effective_DxItem_list,
        )
        self.assertTrue(
            self.support.supports("Histologic_Type", "Invasive Carcinoma(IC)")
        )
        self.assertTrue(self.support.supports("Microcalcification", "Present"))

    def test_fallback_top_k_must_be_positive(self) -> None:
        config = dict(self.config)
        config["fallback_top_k"] = 0
        with self.assertRaisesRegex(ValueError, "fallback_top_k"):
            load_checkpoint_support(config)

    def test_by_dxresult_policy_resolves_threshold_or_records_item_fallback(
        self,
    ) -> None:
        checkpoint_config = json.loads(
            self.support.config_path.read_text(encoding="utf-8")
        )
        threshold_bundle = json.loads(
            self.support.threshold_bundle_path.read_text(encoding="utf-8")
        )
        threshold_bundle["schema_version"] = (
            "clee_evidence_threshold_bundle_v1"
        )
        threshold_bundle["case_importance"]["by_DxResult"] = {
            "Histologic_Type": {
                "Invasive Carcinoma(IC)": {
                    "status": "ok",
                    "threshold": 0.91,
                    "comparison": ">=",
                },
                "Normal(N)": {
                    "status": "insufficient_classes",
                    "threshold": None,
                    "comparison": ">=",
                },
            },
            "Microcalcification": {},
        }

        with tempfile.TemporaryDirectory(
            dir=REFERENCE_ROOT / "person_e"
        ) as directory:
            root = Path(directory)
            config_path = root / "config.json"
            threshold_path = root / "threshold.json"
            config_path.write_text(
                json.dumps(checkpoint_config), encoding="utf-8"
            )
            threshold_path.write_text(
                json.dumps(threshold_bundle), encoding="utf-8"
            )
            config = {
                **self.config,
                "checkpoint_config_path": str(config_path),
                "threshold_bundle_path": str(threshold_path),
                "threshold_policy": "by_DxResult",
            }
            support = load_checkpoint_support(config)

        selected = support.threshold_resolution(
            "Histologic_Type", "Invasive Carcinoma(IC)"
        )
        self.assertEqual("by_DxResult", selected["resolved_policy"])
        self.assertEqual(0.91, selected["threshold"])
        fallback = support.threshold_resolution(
            "Histologic_Type", "Normal(N)"
        )
        self.assertEqual("by_DxItem", fallback["resolved_policy"])
        self.assertEqual("insufficient_classes", fallback["reason"])
        self.assertEqual(
            support.case_importance_thresholds["Histologic_Type"],
            fallback["threshold"],
        )

    def test_multiple_dxitem_traces_merge_on_one_roi(self) -> None:
        roi = {"roi_id": "roi-1"}
        _merge_pseudo_dx_pair(
            roi,
            "Histologic_Type",
            {
                "0": {"Histologic_Type": {"assigned": "Invasive Carcinoma(IC)"}},
                "finalResult": {
                    "Histologic_Type": {"assigned": "Invasive Carcinoma(IC)"}
                },
            },
        )
        _merge_pseudo_dx_pair(
            roi,
            "Microcalcification",
            {
                "0": {"Microcalcification": {"assigned": "Present"}},
                "finalResult": {"Microcalcification": {"assigned": "Present"}},
            },
        )
        self.assertEqual(
            {"Histologic_Type", "Microcalcification"},
            set(roi["pseudo_DxPair"]["finalResult"]),
        )

    def test_fixture_backend_preserves_all_rois_across_hierarchical_chunks(self) -> None:
        payload = {
            "case_list": [
                {
                    "tissue_blocks": [
                        {
                            "stains": [
                                {
                                    "roi_list": [
                                        {"roi_id": f"roi-{idx}", "global_idx": idx}
                                        for idx in range(5)
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        result = run_fixture_backend(
            input_payload=payload,
            DxItem="Histologic_Type",
            case_label="Invasive Carcinoma(IC)",
            support=self.support,
            max_inputs_per_forward=2,
            score_cycle=[0.82, 0.55],
        )

        self.assertEqual(
            {f"roi-{idx}" for idx in range(5)},
            set(result.predictions_by_roi_id),
        )
        for idx in range(5):
            trace = result.predictions_by_roi_id[f"roi-{idx}"]
            self.assertIn("0", trace)
            self.assertIn("finalResult", trace)
            prediction = trace["finalResult"]["Histologic_Type"]
            self.assertEqual(idx % 2 == 0, prediction["assigned_as_ref"])
            if idx % 2 == 0:
                self.assertIn("1", trace)
            else:
                self.assertNotIn("1", trace)

    def test_zero_candidate_fallback_uses_terminal_layer_and_stable_id(self) -> None:
        def prediction(score: float) -> dict[str, object]:
            return {
                "chunk_idx": 0,
                "assigned": "unable_to_determine",
                "importance": score,
                "assigned_as_ref": False,
                "candidate@top3AUC": {},
                "candidate@ALL": {},
            }

        traces = {
            "roi-1": {
                "0": {"Histologic_Type": prediction(0.99)},
                "finalResult": {"Histologic_Type": prediction(0.99)},
            },
            "roi-3": {
                "0": {"Histologic_Type": prediction(0.80)},
                "1": {"Histologic_Type": prediction(0.40)},
                "finalResult": {"Histologic_Type": prediction(0.40)},
            },
            "roi-4": {
                "0": {"Histologic_Type": prediction(0.75)},
                "1": {"Histologic_Type": prediction(0.40)},
                "finalResult": {"Histologic_Type": prediction(0.40)},
            },
        }
        selections, summary = apply_effective_selection(
            traces=traces,
            global_idx_by_roi_id={"roi-1": 1, "roi-3": 3, "roi-4": 4},
            DxItem="Histologic_Type",
            fallback_top_k=1,
        )

        self.assertFalse(
            traces["roi-1"]["finalResult"]["Histologic_Type"][
                "assigned_as_ref"
            ]
        )
        self.assertTrue(
            traces["roi-3"]["finalResult"]["Histologic_Type"][
                "assigned_as_ref"
            ]
        )
        self.assertFalse(
            traces["roi-4"]["finalResult"]["Histologic_Type"][
                "assigned_as_ref"
            ]
        )
        self.assertEqual(
            "zero_candidate_topk_fallback",
            selections["roi-3"]["selection_source"],
        )
        self.assertEqual(1, selections["roi-3"]["fallback_rank"])
        self.assertEqual(1, summary["terminal_layer_idx"])
        self.assertEqual(2, summary["fallback_pool_size"])
        self.assertEqual(0, summary["raw_retained_count"])
        self.assertEqual(1, summary["effective_retained_count"])

    def test_fallback_selection_controls_i_event(self) -> None:
        trace = {
            "0": {
                "Histologic_Type": {
                    "chunk_idx": 0,
                    "assigned": "unable_to_determine",
                    "importance": 0.55,
                    "assigned_as_ref": True,
                    "candidate@top3AUC": {},
                    "candidate@ALL": {},
                }
            },
            "finalResult": {
                "Histologic_Type": {
                    "chunk_idx": 0,
                    "assigned": "unable_to_determine",
                    "importance": 0.55,
                    "assigned_as_ref": True,
                    "candidate@top3AUC": {},
                    "candidate@ALL": {},
                }
            },
        }
        event = _prediction_event(
            config=self.config,
            support=self.support,
            dx_pair_id="dx-1",
            DxItem="Histologic_Type",
            case_result="Invasive Carcinoma(IC)",
            trace=trace,
            selection_detail={
                "raw_retained": False,
                "effective_assigned_as_ref": True,
                "selection_source": "zero_candidate_topk_fallback",
            },
            query_ids=["query-1"],
        )
        self.assertEqual("selected", event["status"])
        self.assertEqual("zero_candidate_topk_fallback", event["reason"])
        self.assertEqual(0.55, event["score"])
        self.assertTrue(event["assigned_as_ref"])


if __name__ == "__main__":
    unittest.main()
