"""Load the externally maintained diagnostic and visual-attribute references."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from components.person_a.asset_integrity import verify_external_asset
from contracts.paths import resolve_person_reference_path, sibling_logical_path


def resolve_reference(configured_path: str | Path) -> Path:
    """Resolve an external Person A asset and enforce reference/person_a/."""

    path = resolve_person_reference_path("person_a", configured_path)
    if not path.is_file():
        raise FileNotFoundError(f"Person A reference file was not found: {path}")
    return path


def load_reference(
    configured_path: str | Path,
    *,
    integrity_mode: str = "enforce",
) -> tuple[Any, dict[str, Any]]:
    path = resolve_reference(configured_path)
    raw = path.read_bytes()
    integrity = verify_external_asset(path, integrity_mode=integrity_mode)
    return json.loads(raw.decode("utf-8")), {
        "source_name": path.name,
        "source_path": sibling_logical_path(path),
        **integrity,
    }


def load_dx_candidates(
    configured_path: str | Path,
    *,
    integrity_mode: str = "enforce",
) -> tuple[dict[str, Any], dict[str, Any]]:
    document, provenance = load_reference(
        configured_path, integrity_mode=integrity_mode
    )
    return document["structured_report"], provenance


def load_histologic_mapping(
    configured_path: str | Path,
    *,
    integrity_mode: str = "enforce",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    document, provenance = load_reference(
        configured_path, integrity_mode=integrity_mode
    )
    mapping = document.get("Histologic_Type_mappingTable")
    if not isinstance(mapping, dict):
        raise ValueError("Histologic Type mapping table must be an object")
    for result_class, entry in mapping.items():
        if not isinstance(result_class, str) or not result_class:
            raise ValueError("Histologic Type mapping keys must be non-empty strings")
        if not isinstance(entry, dict):
            raise ValueError(
                f"Histologic Type mapping {result_class!r} must be an object"
            )
        dx_result = entry.get("DxResult")
        type_level_idx = entry.get("typeLevel_idx")
        if not isinstance(dx_result, str) or not dx_result:
            raise ValueError(
                f"Histologic Type mapping {result_class!r} needs DxResult"
            )
        if not isinstance(type_level_idx, int) or type_level_idx < 0:
            raise ValueError(
                f"Histologic Type mapping {result_class!r} needs a non-negative "
                "typeLevel_idx"
            )
    return mapping, provenance


def load_visual_template(
    configured_path: str | Path,
    *,
    integrity_mode: str = "enforce",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the condition-free visual schema used by inference.

    The inference template is deliberately incapable of carrying subtype-level
    conditions.  Those values must be supplied by the query generator.
    """

    document, provenance = load_reference(
        configured_path, integrity_mode=integrity_mode
    )
    if not isinstance(document, dict):
        raise ValueError("Visual attribute template must be an object")
    if document.get("version") != "1.2.1":
        raise ValueError("Visual attribute template version must be '1.2.1'")
    if document.get("type") != "visualAttributeTemplate":
        raise ValueError(
            "Visual attribute template type must be 'visualAttributeTemplate'"
        )
    condition_labels = document.get("conditionLabels")
    if not isinstance(condition_labels, list) or not condition_labels:
        raise ValueError("Visual attribute template needs conditionLabels")
    if len(condition_labels) != len(set(condition_labels)) or not all(
        isinstance(label, str) and label for label in condition_labels
    ):
        raise ValueError(
            "Visual attribute template conditionLabels must be unique strings"
        )
    visual_attrs = document.get("visualAttrs")
    if not isinstance(visual_attrs, dict) or not visual_attrs:
        raise ValueError("Visual attribute template needs visualAttrs")
    for category, features in visual_attrs.items():
        if not isinstance(features, dict) or not features:
            raise ValueError(
                f"Visual attribute template category {category!r} must be an object"
            )
        for feature, definition in features.items():
            if not isinstance(definition, dict):
                raise ValueError(
                    f"Visual attribute template {category}.{feature} must be an object"
                )
            if "conditions" in definition:
                raise ValueError(
                    f"Inference template must not contain conditions at "
                    f"{category}.{feature}"
                )
            options = definition.get("options")
            option_type = definition.get("type")
            if (
                not isinstance(options, list)
                or not options
                or len(options) != len(set(options))
                or not all(isinstance(option, str) and option for option in options)
            ):
                raise ValueError(
                    f"Visual attribute template {category}.{feature} needs unique "
                    "non-empty options"
                )
            if not isinstance(option_type, str) or not option_type:
                raise ValueError(
                    f"Visual attribute template {category}.{feature} needs type"
                )
    return deepcopy(document), provenance
