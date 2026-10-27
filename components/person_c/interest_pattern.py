"""Person C (丙) interest pattern extraction: CaseList -> E.ROIs.

Two config modes share one CLI:

* ``mode: "example"`` keeps the contract-only stub (no slide is opened, empty
  ROI lists); the shared pipeline/tests use it.
* ``mode: "region_proposal"`` runs the CONCH region-proposal chain on every
  processed stain and converts its polygons to rectangular ROIs
  (one bounding box per connected component; rules: ``components/person_c/rois.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from components.person_c.assets import verify_assets
from components.person_c.rois import (
    build_roi_info,
    candidate_event,
    plan_boxes,
)
from contracts.paths import resolve_person_reference_path
from contracts.runtime import (
    cli_parser,
    load_case_list_input,
    load_config,
    write_artifact,
)

Propose = Callable[[str, Mapping[str, Any]], list[dict[str, Any]]]
SlideInfo = Callable[[str], tuple[tuple[int, int], tuple[float, float]]]


def _producer(config: Mapping[str, Any]) -> str:
    producer = config.get("component_version")
    if not isinstance(producer, str) or not producer:
        raise ValueError("Person C config requires component_version")
    return producer


def _artifact(source: dict[str, Any], case: dict[str, Any], producer: str,
              artifact_id: str, reference_versions: dict[str, Any]) -> dict[str, Any]:
    case_id = case["case_id"]
    return {
        "contract": "E.ROIs",
        "schema_version": "2.0",
        "artifact_id": artifact_id,
        "case_id": case_id,
        "producer": producer,
        "payload": {
            "data_mode": "inference",
            "DxItem_list": list(source["DxItem_list"]),
            "reference_versions": reference_versions,
            "case_list": [case],
        },
    }


def build_example_artifact(source: dict[str, Any], producer: str) -> dict[str, Any]:
    case = deepcopy(source["case_list"][0])
    for block in case["tissue_blocks"]:
        for stain in block["stains"]:
            stain["roi_num"] = 0
            stain["roi_list"] = []
    return _artifact(
        source,
        case,
        producer,
        f"E-example-{case['case_id']}",
        {"example_stub": {"implemented": False, "purpose": "contract smoke test"}},
    )


def _region_cfg(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve assets and check their pinned digests.

    Only ``conch_checkpoint`` is external (below reference/person_c). The CONCH library and
    the BCSS vocabulary ship inside the component; their paths are relative to this package.
    """

    region = dict(config["region"])
    checkpoint = resolve_person_reference_path("person_c", region["conch_checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"person_c reference asset is missing: {checkpoint}")
    region["conch_checkpoint"] = str(checkpoint)
    package = Path(__file__).resolve().parent
    for key in ("conch_lib_path", "vocab_json"):
        path = (package / region[key]).resolve()
        if not path.is_relative_to(package):
            raise ValueError(f"region.{key} must stay inside the component: {path}")
        region[key] = str(path)
    verify_assets(region)
    return region


def slide_info(path: str) -> tuple[tuple[int, int], tuple[float, float]]:
    """Return ``((width, height), (mpp_x, mpp_y))`` of level 0; fail if MPP is absent."""

    import openslide

    if not Path(path).is_file():
        raise FileNotFoundError(f"WSI is missing: {path}")
    with openslide.OpenSlide(path) as slide:
        mpp_x = slide.properties.get("openslide.mpp-x")
        mpp_y = slide.properties.get("openslide.mpp-y")
        if not mpp_x or not mpp_y:
            raise ValueError(f"WSI has no openslide.mpp-x/mpp-y metadata: {path}")
        width, height = slide.level_dimensions[0]
    mpp = (float(mpp_x), float(mpp_y))
    if any(value <= 0 for value in mpp):
        raise ValueError(f"WSI has a non-positive MPP {mpp}: {path}")
    return (int(width), int(height)), mpp


def build_region_artifact(
    source: dict[str, Any],
    config: Mapping[str, Any],
    *,
    propose: Propose,
    slide_info: SlideInfo,
) -> dict[str, Any]:
    producer = _producer(config)
    region = dict(config["region"])
    roi_cfg = config["roi"]
    stain_types = config.get("stain_types")
    case = deepcopy(source["case_list"][0])
    global_idx = 0
    for block in case["tissue_blocks"]:
        for stain in block["stains"]:
            stain["roi_list"] = []
            if stain_types is not None and stain["stain_type"] not in stain_types:
                stain["roi_num"] = 0
                continue
            slide_wh, mpp = slide_info(stain["filepath"])
            boxes = plan_boxes(
                propose(stain["filepath"], region),
                slide_wh,
                float(roi_cfg["min_region_area_px"]),
                int(roi_cfg["max_rois_per_stain"]),
            )
            for local_idx, box in enumerate(boxes):
                level0, main = build_roi_info(
                    box,
                    mpp,
                    float(roi_cfg["target_mpp"]),
                    int(roi_cfg["max_main_side_px"]),
                )
                stain["roi_list"].append(
                    {
                        "roi_id": f"{stain['stain_id']}-roi-{local_idx:05d}",
                        "global_idx": global_idx,
                        "local_idx": local_idx,
                        "level0_info": level0,
                        "main_info": main,
                        "DxPair": None,
                        "visualAttrs": None,
                        "visualAttrs_info": None,
                        "selection_history": [candidate_event(producer)],
                    }
                )
                global_idx += 1
            stain["roi_num"] = len(stain["roi_list"])
    return _artifact(
        source,
        case,
        producer,
        f"E-{case['case_id']}",
        {"interest_pattern": {"region": region, "roi": dict(roi_cfg)}},
    )


def main() -> None:
    args = cli_parser("Person C: CaseList input to E.ROIs").parse_args()
    if len(args.input) != 1:
        raise ValueError("interest_pattern accepts exactly one CaseList input")
    source = load_case_list_input(args.input[0], single_case=True)
    config = load_config(args.config)
    mode = config.get("mode")
    if mode == "example":
        artifact = build_example_artifact(source, _producer(config))
    elif mode == "region_proposal":
        from components.person_c.region.proposal import propose_regions

        resolved_region = _region_cfg(config)
        artifact = build_region_artifact(
            source,
            config,
            propose=lambda path, _region: propose_regions(path, resolved_region),
            slide_info=slide_info,
        )
    else:
        raise ValueError(f"Unsupported person_c mode: {mode!r}")
    write_artifact(artifact, args.output, "E.ROIs")


if __name__ == "__main__":
    main()
