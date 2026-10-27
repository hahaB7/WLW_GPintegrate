"""Thin WLW adapter primitives for a replaceable inference-only CLEE backend."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from contracts.metadata import iter_rois
from contracts.paths import (
    PROJECT_ROOT,
    RUN_ROOT,
    resolve_person_reference_path,
    resolve_run_path,
)


FINAL_RESULT_KEY = "finalResult"
UNABLE_TO_DETERMINE = "unable_to_determine"
EVIDENCE_THRESHOLD_BUNDLE_SCHEMA = "clee_evidence_threshold_bundle_v1"
LEGACY_THRESHOLD_BUNDLE_SCHEMA = "clee_threshold_bundle_v3"
SUPPORTED_THRESHOLD_BUNDLE_SCHEMAS = (
    EVIDENCE_THRESHOLD_BUNDLE_SCHEMA,
    LEGACY_THRESHOLD_BUNDLE_SCHEMA,
)
THRESHOLD_POLICY_BY_DXITEM = "by_DxItem"
THRESHOLD_POLICY_BY_DXRESULT = "by_DxResult"
THRESHOLD_POLICIES = (
    THRESHOLD_POLICY_BY_DXITEM,
    THRESHOLD_POLICY_BY_DXRESULT,
)
SELECTION_SOURCE_THRESHOLD = "threshold"
SELECTION_SOURCE_ZERO_CANDIDATE_FALLBACK = (
    "zero_candidate_topk_fallback"
)
ZERO_CANDIDATE_FALLBACK_POLICY = "terminal_layer_top_k"


@dataclass(frozen=True)
class CheckpointSupport:
    config_path: Path
    model_checkpoint_path: Path | None
    threshold_bundle_path: Path
    active_DxItem_list: tuple[str, ...]
    active_DxResult_dict: dict[str, tuple[str, ...]]
    effective_DxItem_list: tuple[str, ...]
    checkpoint_epoch: int
    bundle_id: str
    threshold_policy: str
    fallback_top_k: int
    case_importance_thresholds: dict[str, float]
    resolved_case_importance_thresholds: dict[
        str, dict[str, dict[str, Any]]
    ]
    roi_classification_thresholds: dict[str, dict[str, float | None]]

    def supports(self, DxItem: str, DxResult: str | None = None) -> bool:
        if DxItem not in self.effective_DxItem_list:
            return False
        if DxResult is None:
            return True
        return DxResult in self.active_DxResult_dict.get(DxItem, ())

    def threshold_resolution(
        self, DxItem: str, DxResult: str
    ) -> dict[str, Any]:
        try:
            return self.resolved_case_importance_thresholds[DxItem][DxResult]
        except KeyError as error:
            raise ValueError(
                f"No resolved CLEE evidence threshold for {DxItem}/{DxResult}"
            ) from error

    def threshold_for(self, DxItem: str, DxResult: str) -> float:
        return float(self.threshold_resolution(DxItem, DxResult)["threshold"])

    @property
    def threshold_resolution_records(self) -> list[dict[str, Any]]:
        return [
            dict(self.resolved_case_importance_thresholds[DxItem][DxResult])
            for DxItem in self.effective_DxItem_list
            for DxResult in self.active_DxResult_dict.get(DxItem, ())
        ]


@dataclass(frozen=True)
class BackendResult:
    predictions_by_roi_id: dict[str, dict[str, Any]]
    execution_status: str
    selection_by_roi_id: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    selection_summary: dict[str, Any] = field(default_factory=dict)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def resolve_project_path(raw_path: str | None) -> Path | None:
    """Resolve a configured CLEE asset below the sibling reference root."""

    if raw_path is None:
        return None
    return resolve_person_reference_path("person_e", raw_path)


def _checkpoint_paths(
    config: Mapping[str, Any],
) -> tuple[Path, Path | None, Path, int | None]:
    checkpoint_dir = resolve_project_path(config.get("checkpoint_dir"))
    model_checkpoint_path: Path | None = None
    checkpoint_epoch: int | None = None
    if checkpoint_dir is not None:
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"CLEE checkpoint directory does not exist: {checkpoint_dir}"
            )
        manifest_path = checkpoint_dir / "[checkpoint]best_model.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"CLEE checkpoint manifest does not exist: {manifest_path}"
            )
        manifest = _load_json(manifest_path)
        metadata = manifest.get("checkpoint_metadata")
        if not isinstance(metadata, Mapping) or metadata.get("epoch_idx") is None:
            raise ValueError(
                f"checkpoint_metadata.epoch_idx is missing in {manifest_path}"
            )
        checkpoint_epoch = int(metadata["epoch_idx"])
        model_filename = manifest.get("model_filename")
        if not isinstance(model_filename, str) or not model_filename:
            raise ValueError(f"model_filename is missing in {manifest_path}")
        model_checkpoint_path = checkpoint_dir / model_filename
        if not model_checkpoint_path.is_file():
            raise FileNotFoundError(
                f"CLEE prefix checkpoint does not exist: {model_checkpoint_path}"
            )

    config_path = resolve_project_path(config.get("checkpoint_config_path"))
    if config_path is None and checkpoint_dir is not None:
        config_path = checkpoint_dir / "config.json"
    if config_path is None or not config_path.is_file():
        raise FileNotFoundError(
            "CLEE checkpoint config is required and must exist: "
            f"{config_path}"
        )

    threshold_path = resolve_project_path(config.get("threshold_bundle_path"))
    if threshold_path is None:
        if checkpoint_dir is None:
            raise ValueError(
                "threshold_bundle_path is required when checkpoint_dir is absent"
            )
        epoch_prefix = f"threshold_bundle[{checkpoint_epoch:03d}]_"
        matches = sorted(
            path
            for path in checkpoint_dir.iterdir()
            if path.name.startswith(epoch_prefix) and path.suffix == ".json"
        )
        if len(matches) != 1:
            raise ValueError(
                "Expected one threshold bundle for checkpoint epoch "
                f"{checkpoint_epoch}, "
                f"found {len(matches)}"
            )
        threshold_path = matches[0]
    if not threshold_path.is_file():
        raise FileNotFoundError(f"CLEE threshold bundle does not exist: {threshold_path}")
    return (
        config_path,
        model_checkpoint_path,
        threshold_path,
        checkpoint_epoch,
    )


def _prefix_label_space(
    checkpoint_config: Mapping[str, Any],
    model_checkpoint_path: Path | None,
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    def normalized(
        source: Mapping[str, Any],
        description: str,
    ) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
        raw_items = source.get("active_DxItem_list")
        raw_results = source.get("active_DxResult_dict")
        if (
            not isinstance(raw_items, Sequence)
            or isinstance(raw_items, (str, bytes))
            or not isinstance(raw_results, Mapping)
        ):
            raise ValueError(
                f"{description} has no usable active CLEE label space"
            )
        active_items = tuple(str(item) for item in raw_items)
        active_results = {
            str(item): tuple(str(result) for result in results)
            for item, results in raw_results.items()
            if isinstance(results, Sequence)
            and not isinstance(results, (str, bytes))
        }
        if not active_items or len(set(active_items)) != len(active_items):
            raise ValueError(
                f"{description} active_DxItem_list must be non-empty and unique"
            )
        if set(active_items) != set(active_results):
            raise ValueError(
                f"{description} active DxItems and active DxResults disagree"
            )
        if any(
            not results or len(set(results)) != len(results)
            for results in active_results.values()
        ):
            raise ValueError(
                f"{description} active_DxResult_dict must contain unique labels"
            )
        return active_items, active_results

    config_space = normalized(checkpoint_config, "CLEE checkpoint config")
    if model_checkpoint_path is None:
        return config_space

    import torch

    try:
        payload = torch.load(
            model_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(model_checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(
            "CLEE prefix checkpoint must contain an object payload: "
            f"{model_checkpoint_path}"
        )
    prefix_space = normalized(payload, "CLEE prefix checkpoint")
    if config_space != prefix_space:
        raise ValueError(
            "CLEE checkpoint config/prefix active label-space mismatch"
        )

    configured_items = checkpoint_config.get("DxItem_list")
    prefix_items = payload.get("DxItem_list")
    if (
        isinstance(configured_items, Sequence)
        and not isinstance(configured_items, (str, bytes))
        and isinstance(prefix_items, Sequence)
        and not isinstance(prefix_items, (str, bytes))
        and tuple(str(item) for item in configured_items)
        != tuple(str(item) for item in prefix_items)
    ):
        raise ValueError(
            "CLEE checkpoint config/prefix declared DxItem_list mismatch"
        )
    return prefix_space


def _usable_threshold(
    threshold_info: Any,
    *,
    description: str,
    required: bool,
) -> float | None:
    if not isinstance(threshold_info, Mapping):
        if required:
            raise ValueError(f"Missing threshold information for {description}")
        return None
    threshold = threshold_info.get("threshold")
    if threshold is None:
        if required:
            raise ValueError(f"Missing threshold for {description}")
        return None
    if threshold_info.get("comparison") != ">=":
        raise ValueError(f"Unsupported threshold comparison for {description}")
    value = float(threshold)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite threshold for {description}")
    return value


def load_checkpoint_support(config: Mapping[str, Any]) -> CheckpointSupport:
    """Read checkpoint-declared support without importing the CLEE runtime."""

    (
        config_path,
        model_checkpoint_path,
        threshold_path,
        manifest_epoch,
    ) = _checkpoint_paths(config)
    checkpoint_config = _load_json(config_path)
    threshold_bundle = _load_json(threshold_path)

    if threshold_bundle.get("schema_version") not in (
        SUPPORTED_THRESHOLD_BUNDLE_SCHEMAS
    ):
        raise ValueError(
            "Unsupported CLEE threshold schema: "
            f"{threshold_bundle.get('schema_version')!r}"
        )
    if threshold_bundle.get("source_split") != "valid":
        raise ValueError("CLEE thresholds must be calibrated from the valid split")
    if threshold_bundle.get("score_type") != "cosine_similarity":
        raise ValueError("CLEE selection requires cosine_similarity thresholds")
    threshold_epoch = int(threshold_bundle.get("epoch_idx", -1))
    if manifest_epoch is not None and threshold_epoch != manifest_epoch:
        raise ValueError(
            "CLEE checkpoint/threshold epoch mismatch: "
            f"{manifest_epoch} != {threshold_epoch}"
        )

    config_active_items, config_active_results = _prefix_label_space(
        checkpoint_config,
        model_checkpoint_path,
    )
    label_space = threshold_bundle.get("label_space", {})
    threshold_active_items = tuple(
        str(item) for item in label_space.get("active_DxItem_list", [])
    )
    threshold_active_results = {
        str(item): tuple(str(result) for result in results)
        for item, results in label_space.get("active_DxResult_dict", {}).items()
    }
    if config_active_items != threshold_active_items:
        raise ValueError(
            "CLEE config/threshold active_DxItem_list mismatch: "
            f"config={config_active_items}, threshold={threshold_active_items}"
        )
    if config_active_results != threshold_active_results:
        raise ValueError("CLEE config/threshold active_DxResult_dict mismatch")

    raw_policy = config.get("wlw_supported_dx_items")
    if raw_policy is None:
        # The checkpoint and its validation-calibrated threshold bundle are the
        # authoritative declaration of the label space.  A deployment may
        # still provide an explicit list to narrow (never expand) that space.
        effective_items = config_active_items
    elif (
        isinstance(raw_policy, Sequence)
        and not isinstance(raw_policy, (str, bytes))
        and all(isinstance(item, str) and item for item in raw_policy)
    ):
        wlw_whitelist = tuple(dict.fromkeys(str(item) for item in raw_policy))
        unknown_items = set(wlw_whitelist) - set(config_active_items)
        if unknown_items:
            raise ValueError(
                "wlw_supported_dx_items cannot expand checkpoint support: "
                f"{sorted(unknown_items)}"
            )
        effective_items = tuple(
            item for item in wlw_whitelist if item in config_active_items
        )
    else:
        raise ValueError(
            "wlw_supported_dx_items must be null or a list of nonempty strings"
        )
    if not effective_items:
        raise ValueError("No effective CLEE DxItems remain after deployment policy")

    threshold_policy = str(
        config.get("threshold_policy", THRESHOLD_POLICY_BY_DXITEM)
    )
    if threshold_policy not in THRESHOLD_POLICIES:
        raise ValueError(
            f"threshold_policy must be one of {THRESHOLD_POLICIES!r}"
        )
    fallback_top_k = config.get("fallback_top_k", 1)
    if (
        isinstance(fallback_top_k, bool)
        or not isinstance(fallback_top_k, int)
        or fallback_top_k <= 0
    ):
        raise ValueError("fallback_top_k must be a positive integer")

    case_importance = threshold_bundle.get("case_importance", {})
    if not isinstance(case_importance, Mapping):
        raise ValueError("CLEE case_importance must be an object")
    if case_importance.get("calibration_level") != "DxItem":
        raise ValueError("CLEE case_importance calibration_level must be DxItem")
    by_DxItem = case_importance.get("by_DxItem", {})
    if not isinstance(by_DxItem, Mapping):
        raise ValueError("CLEE case_importance.by_DxItem must be an object")
    by_DxResult = case_importance.get("by_DxResult", {})
    if threshold_policy == THRESHOLD_POLICY_BY_DXRESULT and not isinstance(
        by_DxResult, Mapping
    ):
        raise ValueError(
            "by_DxResult policy requires case_importance.by_DxResult"
        )
    if not isinstance(by_DxResult, Mapping):
        by_DxResult = {}

    case_thresholds: dict[str, float] = {}
    resolved_thresholds: dict[str, dict[str, dict[str, Any]]] = {}
    roi_thresholds: dict[str, dict[str, float | None]] = {}
    all_roi_thresholds = threshold_bundle.get("roi_classification", {})
    for DxItem in effective_items:
        item_threshold = _usable_threshold(
            by_DxItem.get(DxItem),
            description=f"case_importance.by_DxItem[{DxItem!r}]",
            required=True,
        )
        assert item_threshold is not None
        case_thresholds[DxItem] = item_threshold
        resolved_thresholds[DxItem] = {}
        item_result_thresholds = by_DxResult.get(DxItem, {})
        if item_result_thresholds is None:
            item_result_thresholds = {}
        if not isinstance(item_result_thresholds, Mapping):
            raise ValueError(
                f"case_importance.by_DxResult[{DxItem!r}] must be an object"
            )
        for DxResult in config_active_results.get(DxItem, ()):
            threshold = item_threshold
            resolved_policy = THRESHOLD_POLICY_BY_DXITEM
            reason = "selected_policy"
            if threshold_policy == THRESHOLD_POLICY_BY_DXRESULT:
                result_info = item_result_thresholds.get(DxResult)
                status = (
                    str(result_info.get("status"))
                    if isinstance(result_info, Mapping)
                    and result_info.get("status")
                    else None
                )
                result_threshold = (
                    None
                    if status not in (None, "ok")
                    else _usable_threshold(
                        result_info,
                        description=(
                            "case_importance.by_DxResult"
                            f"[{DxItem!r}][{DxResult!r}]"
                        ),
                        required=False,
                    )
                )
                if result_threshold is not None:
                    threshold = result_threshold
                    resolved_policy = THRESHOLD_POLICY_BY_DXRESULT
                    reason = "checkpoint_seen_with_calibrated_threshold"
                else:
                    reason = (
                        status
                        if status not in (None, "ok")
                        else "missing_case_importance_by_DxResult_threshold"
                    )
            resolved_thresholds[DxItem][DxResult] = {
                "DxItem": DxItem,
                "DxResult": DxResult,
                "checkpoint_seen": True,
                "requested_policy": threshold_policy,
                "resolved_policy": resolved_policy,
                "reason": reason,
                "threshold": threshold,
                "comparison": ">=",
            }

        item_thresholds = all_roi_thresholds.get(DxItem)
        if not isinstance(item_thresholds, Mapping):
            raise ValueError(f"Missing CLEE ROI thresholds for {DxItem}")
        roi_thresholds[DxItem] = {}
        for DxResult in config_active_results.get(DxItem, ()):
            result_info = item_thresholds.get(DxResult)
            if not isinstance(result_info, Mapping):
                raise ValueError(f"Missing CLEE ROI threshold for {DxItem}/{DxResult}")
            if result_info.get("comparison") != ">=":
                raise ValueError(
                    f"Unsupported CLEE ROI threshold comparison for "
                    f"{DxItem}/{DxResult}"
                )
            raw_threshold = result_info.get("threshold")
            roi_thresholds[DxItem][DxResult] = (
                None if raw_threshold is None else float(raw_threshold)
            )

    return CheckpointSupport(
        config_path=config_path,
        model_checkpoint_path=model_checkpoint_path,
        threshold_bundle_path=threshold_path,
        active_DxItem_list=config_active_items,
        active_DxResult_dict=config_active_results,
        effective_DxItem_list=effective_items,
        checkpoint_epoch=threshold_epoch,
        bundle_id=str(threshold_bundle.get("bundle_id") or "unidentified-bundle"),
        threshold_policy=threshold_policy,
        fallback_top_k=int(fallback_top_k),
        case_importance_thresholds=case_thresholds,
        resolved_case_importance_thresholds=resolved_thresholds,
        roi_classification_thresholds=roi_thresholds,
    )


def _stable_global_idx_key(global_idx: Any) -> tuple[str, Any]:
    if isinstance(global_idx, bool):
        return ("bool", int(global_idx))
    if isinstance(global_idx, int):
        return ("int", global_idx)
    if isinstance(global_idx, float):
        if not math.isfinite(global_idx):
            raise ValueError("ROI global_idx must be finite")
        return ("float", global_idx)
    if isinstance(global_idx, str):
        return ("str", global_idx)
    raise ValueError(f"Unsupported ROI global_idx: {global_idx!r}")


def apply_effective_selection(
    *,
    traces: Mapping[str, dict[str, Any]],
    global_idx_by_roi_id: Mapping[str, Any],
    DxItem: str,
    fallback_top_k: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Apply terminal-layer fallback without losing raw threshold decisions."""

    if (
        isinstance(fallback_top_k, bool)
        or not isinstance(fallback_top_k, int)
        or fallback_top_k <= 0
    ):
        raise ValueError("fallback_top_k must be a positive integer")

    records: list[dict[str, Any]] = []
    for roi_id, trace in traces.items():
        final = trace.get(FINAL_RESULT_KEY)
        prediction = final.get(DxItem) if isinstance(final, Mapping) else None
        if not isinstance(prediction, Mapping):
            raise ValueError(f"CLEE trace for {roi_id}/{DxItem} has no finalResult")
        raw_retained = prediction.get("assigned_as_ref")
        if not isinstance(raw_retained, bool):
            raise ValueError(
                f"CLEE trace for {roi_id}/{DxItem} has no raw decision"
            )
        evidence_score = float(prediction["importance"])
        if not math.isfinite(evidence_score):
            raise ValueError(f"CLEE evidence score for {roi_id}/{DxItem} is not finite")
        numeric_layers = sorted(
            int(layer_key)
            for layer_key, layer in trace.items()
            if layer_key != FINAL_RESULT_KEY
            and str(layer_key).isdigit()
            and isinstance(layer, Mapping)
            and DxItem in layer
        )
        if not numeric_layers:
            raise ValueError(f"CLEE trace for {roi_id}/{DxItem} has no numeric layer")
        records.append({
            "roi_id": roi_id,
            "global_idx": global_idx_by_roi_id[roi_id],
            "raw_retained": raw_retained,
            "evidence_score": evidence_score,
            "last_layer_idx": numeric_layers[-1],
        })

    raw_selected = [record for record in records if record["raw_retained"]]
    terminal_layer_idx: int | None = None
    fallback_pool: list[dict[str, Any]] = []
    fallback_selected: list[dict[str, Any]] = []
    if records and not raw_selected:
        terminal_layer_idx = max(record["last_layer_idx"] for record in records)
        fallback_pool = [
            record
            for record in records
            if record["last_layer_idx"] == terminal_layer_idx
        ]
        fallback_pool.sort(key=lambda record: (
            -record["evidence_score"],
            _stable_global_idx_key(record["global_idx"]),
        ))
        fallback_selected = fallback_pool[:fallback_top_k]

    fallback_rank_by_roi_id = {
        record["roi_id"]: rank
        for rank, record in enumerate(fallback_selected, start=1)
    }
    selection_by_roi_id: dict[str, dict[str, Any]] = {}
    for record in records:
        roi_id = record["roi_id"]
        fallback_rank = fallback_rank_by_roi_id.get(roi_id)
        effective = bool(record["raw_retained"] or fallback_rank is not None)
        detail: dict[str, Any] = {
            "DxItem": DxItem,
            "global_idx": record["global_idx"],
            "raw_retained": bool(record["raw_retained"]),
            "effective_assigned_as_ref": effective,
            "evidence_score": float(record["evidence_score"]),
            "selection_source": SELECTION_SOURCE_THRESHOLD,
            "last_layer_idx": int(record["last_layer_idx"]),
        }
        if fallback_rank is not None:
            detail.update({
                "selection_source": (
                    SELECTION_SOURCE_ZERO_CANDIDATE_FALLBACK
                ),
                "fallback_rank": int(fallback_rank),
                "fallback_top_k": int(fallback_top_k),
                "fallback_layer_idx": int(terminal_layer_idx),
                "fallback_pool_size": int(len(fallback_pool)),
            })
        selection_by_roi_id[roi_id] = detail

        targets = (
            traces[roi_id][str(record["last_layer_idx"])][DxItem],
            traces[roi_id][FINAL_RESULT_KEY][DxItem],
        )
        for target in targets:
            target["assigned_as_ref"] = effective

    summary = {
        "scope": "case_x_DxItem",
        "DxItem": DxItem,
        "fallback_policy": ZERO_CANDIDATE_FALLBACK_POLICY,
        "fallback_top_k": int(fallback_top_k),
        "candidate_assignment_count": len(records),
        "raw_retained_count": len(raw_selected),
        "effective_retained_count": len(raw_selected) + len(fallback_selected),
        "fallback_applied": bool(fallback_selected),
        "fallback_reason": (
            "raw_retained_count_zero"
            if fallback_selected
            else ("no_candidate_assignments" if not records else None)
        ),
        "terminal_layer_idx": terminal_layer_idx,
        "fallback_pool_size": len(fallback_pool),
        "fallback_selections": [
            {
                "roi_id": record["roi_id"],
                "global_idx": record["global_idx"],
                "rank": rank,
                "evidence_score": float(record["evidence_score"]),
            }
            for rank, record in enumerate(fallback_selected, start=1)
        ],
    }
    return selection_by_roi_id, summary


def _fixture_scores(
    labels: Sequence[str],
    case_label: str,
    global_idx: int,
    score_cycle: Sequence[float],
) -> dict[str, float]:
    importance = float(score_cycle[global_idx % len(score_cycle)])
    scores = {
        label: round(max(-1.0, importance - 0.35 - index * 0.01), 6)
        for index, label in enumerate(labels)
    }
    scores[case_label] = round(importance, 6)
    return scores


def _fixture_prediction(
    DxItem: str,
    case_label: str,
    global_idx: int,
    chunk_idx: int,
    support: CheckpointSupport,
    score_cycle: Sequence[float],
) -> dict[str, Any]:
    labels = support.active_DxResult_dict[DxItem]
    all_scores = _fixture_scores(labels, case_label, global_idx, score_cycle)
    candidate_pairs = []
    for label in labels:
        threshold = support.roi_classification_thresholds[DxItem][label]
        if threshold is not None and all_scores[label] >= threshold:
            candidate_pairs.append((label, all_scores[label]))
    candidate_pairs.sort(key=lambda item: item[1], reverse=True)
    assigned = candidate_pairs[0][0] if candidate_pairs else UNABLE_TO_DETERMINE
    importance = all_scores[case_label]
    assigned_as_ref = importance >= support.threshold_for(DxItem, case_label)
    return {
        "chunk_idx": chunk_idx,
        "assigned": assigned,
        "importance": importance,
        "assigned_as_ref": assigned_as_ref,
        "candidate@top3AUC": dict(candidate_pairs[:3]),
        "candidate@ALL": all_scores,
    }


def run_fixture_backend(
    input_payload: dict[str, Any],
    DxItem: str,
    case_label: str,
    support: CheckpointSupport,
    max_inputs_per_forward: int,
    score_cycle: Sequence[float],
) -> BackendResult:
    """Produce CLEE-shaped traces for contract/E2E tests only."""

    if max_inputs_per_forward <= 0:
        raise ValueError("max_roi_dxitem_inputs_per_forward must be positive")
    if not score_cycle:
        raise ValueError("fixture_importance_score_cycle must not be empty")

    roi_records = [
        (roi["roi_id"], int(roi["global_idx"]))
        for _, roi in iter_rois(input_payload)
    ]
    traces: dict[str, dict[str, Any]] = {roi_id: {} for roi_id, _ in roi_records}
    pending = list(roi_records)
    layer_idx = 0
    while pending:
        selected: list[tuple[str, int]] = []
        chunks = [
            pending[index : index + max_inputs_per_forward]
            for index in range(0, len(pending), max_inputs_per_forward)
        ]
        for chunk_idx, chunk in enumerate(chunks):
            for roi_id, global_idx in chunk:
                prediction = _fixture_prediction(
                    DxItem=DxItem,
                    case_label=case_label,
                    global_idx=global_idx,
                    chunk_idx=chunk_idx,
                    support=support,
                    score_cycle=score_cycle,
                )
                traces[roi_id][str(layer_idx)] = {DxItem: prediction}
                traces[roi_id][FINAL_RESULT_KEY] = {
                    DxItem: deepcopy(prediction)
                }
                if prediction["assigned_as_ref"] is True:
                    selected.append((roi_id, global_idx))

        if len(chunks) == 1 or not selected or len(selected) == len(pending):
            break
        pending = selected
        layer_idx += 1

    selection_by_roi_id, selection_summary = apply_effective_selection(
        traces=traces,
        global_idx_by_roi_id=dict(roi_records),
        DxItem=DxItem,
        fallback_top_k=support.fallback_top_k,
    )

    return BackendResult(
        predictions_by_roi_id=traces,
        execution_status="fixture_completed",
        selection_by_roi_id=selection_by_roi_id,
        selection_summary=selection_summary,
    )


def _format_command(
    template: Sequence[str],
    replacements: Mapping[str, str],
) -> list[str]:
    if not template:
        raise ValueError("external_command backend requires command_template")
    return [str(part).format_map(replacements) for part in template]


def run_external_backend(
    input_payload: dict[str, Any],
    DxItem: str,
    support: CheckpointSupport,
    config: Mapping[str, Any],
) -> BackendResult:
    """Invoke an inference-only CLEE CLI and read its metadata output."""

    backend = config["backend"]
    work_root = RUN_ROOT / "output" / "work"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="wlw-clee-", dir=work_root
    ) as temp_dir:
        temporary_dir = Path(temp_dir)
        input_path = temporary_dir / "input_metadata.json"
        output_path = temporary_dir / "output_metadata.json"
        input_path.write_text(
            json.dumps(input_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        replacements = {
            "input_metadata": str(input_path),
            "output_metadata": str(output_path),
            "output_dir": str(temporary_dir),
            "checkpoint_dir": str(
                resolve_project_path(config.get("checkpoint_dir")) or ""
            ),
            "image_root": str(
                resolve_run_path(config["image_root"])
                if config.get("image_root")
                else ""
            ),
            "weight_path": str(
                resolve_project_path(config.get("model_weight_path")) or ""
            ),
            "max_roi_dxitem_inputs_per_forward": str(
                int(config["max_roi_dxitem_inputs_per_forward"])
            ),
            "threshold_policy": support.threshold_policy,
            "fallback_top_k": str(support.fallback_top_k),
        }
        command = _format_command(backend.get("command_template", []), replacements)
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if not output_path.is_file():
            raise FileNotFoundError(
                "Inference-only CLEE did not create its declared output: "
                f"{output_path}"
            )
        output_metadata = _load_json(output_path)

    predictions: dict[str, dict[str, Any]] = {}
    global_idx_by_roi_id: dict[str, Any] = {}
    for _, roi in iter_rois(output_metadata):
        roi_id = str(roi.get("roi_id") or "")
        pseudo_DxPair = roi.get("pseudo_DxPair")
        if not roi_id or not isinstance(pseudo_DxPair, dict):
            raise ValueError(
                "Inference-only CLEE output must preserve roi_id and add "
                "pseudo_DxPair to every submitted ROI"
            )
        predictions[roi_id] = pseudo_DxPair
        global_idx_by_roi_id[roi_id] = roi["global_idx"]
    selection_by_roi_id, selection_summary = apply_effective_selection(
        traces=predictions,
        global_idx_by_roi_id=global_idx_by_roi_id,
        DxItem=DxItem,
        fallback_top_k=support.fallback_top_k,
    )
    return BackendResult(
        predictions_by_roi_id=predictions,
        execution_status="external_completed",
        selection_by_roi_id=selection_by_roi_id,
        selection_summary=selection_summary,
    )


def run_backend(
    input_payload: dict[str, Any],
    DxItem: str,
    case_label: str,
    support: CheckpointSupport,
    config: Mapping[str, Any],
) -> BackendResult:
    mode = str(config["backend"]["mode"])
    if mode == "fixture":
        return run_fixture_backend(
            input_payload=input_payload,
            DxItem=DxItem,
            case_label=case_label,
            support=support,
            max_inputs_per_forward=int(
                config["max_roi_dxitem_inputs_per_forward"]
            ),
            score_cycle=[
                float(value)
                for value in config["backend"]["fixture_importance_score_cycle"]
            ],
        )
    if mode == "external_command":
        return run_external_backend(
            input_payload=input_payload,
            DxItem=DxItem,
            support=support,
            config=config,
        )
    if mode == "native":
        # Keep torch/transformers/OpenSlide imports out of fixture and contract
        # tests; the production dependencies are loaded only for real inference.
        from components.person_e.native_inference import run_native_inference

        result = run_native_inference(
            input_payload=input_payload,
            DxItem=DxItem,
            case_label=case_label,
            support=support,
            config=config,
        )
        return BackendResult(
            predictions_by_roi_id=result.predictions_by_roi_id,
            execution_status="native_completed",
            selection_by_roi_id=result.selection_by_roi_id,
            selection_summary=result.selection_summary,
        )
    if mode == "deferred":
        return BackendResult(
            predictions_by_roi_id={},
            execution_status="backend_deferred",
        )
    raise ValueError(f"Unsupported CLEE backend mode: {mode!r}")
