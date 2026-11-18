from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from components.person_a.reference_data import (
    load_histologic_mapping,
    load_visual_template,
)
from contracts.paths import REFERENCE_ROOT


PERSON_A_REFERENCE = REFERENCE_ROOT / "person_a/template_ref"


class ReferenceDataTests(unittest.TestCase):
    def test_mapping_is_loaded_from_the_configured_file_at_runtime(self) -> None:
        with tempfile.TemporaryDirectory(dir=PERSON_A_REFERENCE) as temp_dir:
            path = Path(temp_dir) / "mapping.json"
            path.write_text(
                json.dumps(
                    {
                        "Histologic_Type_mappingTable": {
                            "Custom": {"DxResult": "Target", "typeLevel_idx": 9}
                        }
                    }
                ),
                encoding="utf-8",
            )
            mapping, provenance = load_histologic_mapping(
                path, integrity_mode="record_only"
            )
            self.assertEqual(
                {"Custom": {"DxResult": "Target", "typeLevel_idx": 9}},
                mapping,
            )
            self.assertEqual("mapping.json", provenance["source_name"])
            self.assertEqual("record_only", provenance["integrity_mode"])

    def test_inference_template_is_condition_free_and_integrity_verified(self) -> None:
        path = PERSON_A_REFERENCE / "visual_attribute_template_v1.2.1.json"
        template, provenance = load_visual_template(path)
        self.assertTrue(provenance["integrity_verified"])
        self.assertEqual("1.2.1", template["version"])
        self.assertEqual("visualAttributeTemplate", template["type"])
        self.assertEqual(
            "ordinal",
            template["visualAttrs"]["Cellular_and_Nuclear"]["Polarity"]["type"],
        )
        for features in template["visualAttrs"].values():
            for definition in features.values():
                self.assertNotIn("conditions", definition)


if __name__ == "__main__":
    unittest.main()
