from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from components.person_e.adapter import (
    FINAL_RESULT_KEY,
    BackendResult,
    CheckpointSupport,
    load_checkpoint_support,
    run_backend,
)
from contracts.metadata import ensure_same_case, iter_rois, single_case
from contracts.paths import sibling_logical_path
from contracts.runtime import cli_parser, load_config, load_inputs, write_artifact


def _mapped_case_result(
    record: Mapping[str, Any],
    DxItem: str,
    support: CheckpointSupport,
) -> tuple[str | None, str]:
    raw_result = record.get("DxResultCls")
    if isinstance(raw_result, str) and support.supports(DxItem, raw_result):
        return raw_result, "structured_report.DxResultCls"

    mapped_results = set()
    for query in record.get("visualAttrQueries", []):
        criteria = query.get("diagnosticCriteria")
        if (
            query.get("criteria_status") == "mapped"
            and isinstance(criteria, Mapping)
            and isinstance(criteria.get("DxResult"), str)
            and support.supports(DxItem, str(criteria["DxResult"]))
        ):
            mapped_results.add(str(criteria["DxResult"]))
    if len(mapped_results) == 1:
        return next(iter(mapped_results)), "visualAttrQueries.diagnosticCriteria.DxResult"
    if len(mapped_results) > 1:
        raise ValueError(
            f"DxPair {record.get('dx_pair_id')} maps to multiple CLEE results: "
            f"{sorted(mapped_results)}"
        )
    return None, "unresolved"


def _matching_events(
    roi: Mapping[str, Any],
    dx_pair_id: str,
) -> list[Mapping[str, Any]]:
    return [
        event
        for event in roi["selection_history"]
        if event["stage"] == "visual_attributes_matching_filter"
        and event.get("dx_pair_id") == dx_pair_id
    ]


def _clee_event(
    *,
    config: Mapping[str, Any],
    support: CheckpointSupport,
    dx_pair_id: str | None,
    action: str,
    status: str,
    reason: str,
    query_ids: list[str] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "stage": "clee",
        "owner": "person_E",
        "artifact_contract": "I.CLEESelectedROIs",
        "action": action,
        "status": status,
        "selected": status == "selected",
        "reason": reason,
        "producer": config["component_version"],
        "backend": config["backend"]["mode"],
        "checkpoint_epoch": support.checkpoint_epoch,
        "bundle_id": support.bundle_id,
        "support_source": sibling_logical_path(support.config_path),
    }
    if dx_pair_id is not None:
        event["dx_pair_id"] = dx_pair_id
    if query_ids:
        event["upstream_query_ids"] = sorted(set(query_ids))
    return event


def _inference_payload(
    source_payload: dict[str, Any],
    eligible_roi_ids: set[str],
    DxItem: str,
    case_result: str,
) -> dict[str, Any]:
    payload = deepcopy(source_payload)
    payload["DxItem_list"] = [DxItem]
    case = payload["case_list"][0]
    item_record = deepcopy(case["structured_report"]["DxItems"][DxItem])
    item_record["DxResultCls"] = case_result
    case["structured_report"]["DxItems"] = {DxItem: item_record}
    for block in case["tissue_blocks"]:
        for stain in block["stains"]:
            stain["roi_list"] = [
                roi
                for roi in stain["roi_list"]
                if roi["roi_id"] in eligible_roi_ids
            ]
            stain["roi_num"] = len(stain["roi_list"])
            for roi in stain["roi_list"]:
                roi.pop("pseudo_DxPair", None)
    return payload


def _merge_pseudo_dx_pair(
    roi: dict[str, Any], DxItem: str, trace: Mapping[str, Any]
) -> None:
    """Merge one DxItem trace without discarding earlier CLEE predictions."""

    target = roi.setdefault("pseudo_DxPair", {})
    if not isinstance(target, dict):
        raise ValueError(f"ROI {roi.get('roi_id')} has an invalid pseudo_DxPair")
    for layer_key, raw_predictions in trace.items():
        if not isinstance(raw_predictions, Mapping) or DxItem not in raw_predictions:
            raise ValueError(
                f"CLEE trace for ROI {roi.get('roi_id')}/{DxItem} has an invalid "
                f"layer {layer_key!r}"
            )
        layer = target.setdefault(str(layer_key), {})
        if not isinstance(layer, dict):
            raise ValueError(
                f"ROI {roi.get('roi_id')} pseudo_DxPair layer {layer_key!r} is invalid"
            )
        if DxItem in layer:
            raise ValueError(
                f"ROI {roi.get('roi_id')} already contains a {DxItem} CLEE prediction"
            )
        layer[DxItem] = deepcopy(raw_predictions[DxItem])


def _prediction_event(
    *,
    config: Mapping[str, Any],
    support: CheckpointSupport,
    dx_pair_id: str,
    DxItem: str,
    case_result: str,
    trace: Mapping[str, Any],
    selection_detail: Mapping[str, Any] | None,
    query_ids: list[str],
) -> dict[str, Any]:
    final_predictions = trace.get(FINAL_RESULT_KEY)
    if not isinstance(final_predictions, Mapping):
        raise ValueError(f"CLEE prediction for {dx_pair_id} has no finalResult")
    prediction = final_predictions.get(DxItem)
    if not isinstance(prediction, Mapping):
        raise ValueError(
            f"CLEE finalResult for {dx_pair_id} has no {DxItem} prediction"
        )
    assigned_as_ref = prediction.get("assigned_as_ref")
    if not isinstance(assigned_as_ref, bool):
        raise ValueError(
            f"CLEE finalResult for {dx_pair_id}/{DxItem} must decide assigned_as_ref"
        )
    importance = prediction.get("importance")
    if not isinstance(importance, (int, float)) or isinstance(importance, bool):
        raise ValueError(
            f"CLEE finalResult for {dx_pair_id}/{DxItem} requires a numeric "
            "evidence score in importance"
        )
    evidence_score = float(importance)
    threshold = support.threshold_for(DxItem, case_result)
    if selection_detail is None:
        raw_retained = evidence_score >= threshold
        selection_source = (
            "zero_candidate_topk_fallback"
            if assigned_as_ref and not raw_retained
            else "threshold"
        )
    else:
        raw_retained = selection_detail.get("raw_retained")
        selection_source = selection_detail.get("selection_source")
        if not isinstance(raw_retained, bool):
            raise ValueError(
                f"CLEE selection audit for {dx_pair_id}/{DxItem} has no raw decision"
            )
        if selection_detail.get("effective_assigned_as_ref") is not assigned_as_ref:
            raise ValueError(
                f"CLEE selection audit for {dx_pair_id}/{DxItem} disagrees "
                "with assigned_as_ref"
            )

    if selection_source == "zero_candidate_topk_fallback":
        reason = "zero_candidate_topk_fallback"
    elif raw_retained:
        reason = "case_importance_threshold_met"
    else:
        reason = "case_importance_threshold_not_met"

    numeric_layers = sorted(
        int(layer_key)
        for layer_key in trace
        if layer_key != FINAL_RESULT_KEY and str(layer_key).isdigit()
    )
    if not numeric_layers:
        raise ValueError(f"CLEE prediction for {dx_pair_id} has no numeric layer")

    event = _clee_event(
        config=config,
        support=support,
        dx_pair_id=dx_pair_id,
        action="evidence_evaluated",
        status="selected" if assigned_as_ref else "rejected",
        reason=reason,
        query_ids=query_ids,
    )
    event.update(
        {
            "threshold": threshold,
            "comparison": ">=",
            "assigned": str(prediction["assigned"]),
            "assigned_as_ref": assigned_as_ref,
            "layer_idx": numeric_layers[-1],
            "chunk_idx": int(prediction["chunk_idx"]),
        }
    )
    event["score"] = evidence_score
    return event


def _record_support(
    DxItem: str,
    record: Mapping[str, Any],
    support: CheckpointSupport,
) -> tuple[str | None, str | None, str]:
    if DxItem not in support.effective_DxItem_list:
        return None, "unsupported_dx_item", "unresolved"
    case_result, source = _mapped_case_result(record, DxItem, support)
    if case_result is None:
        return None, "unsupported_dx_result", source
    return case_result, None, source


def main() -> None:
    args = cli_parser(
        "戊: run or adapt inference-only CLEE and append auditable ROI decisions"
    ).parse_args()
    inputs = load_inputs(args.input, ["D.DxPairs", "H.MatchedROIs"])
    d_artifact = inputs["D.DxPairs"]
    h_artifact = inputs["H.MatchedROIs"]
    config = load_config(args.config)
    case_id = ensure_same_case(d_artifact, h_artifact)
    d_case = single_case(d_artifact)
    h_case = single_case(h_artifact)
    payload = deepcopy(h_artifact["payload"])
    support = load_checkpoint_support(config)

    d_records = d_case["structured_report"]["DxItems"]
    h_records = h_case["structured_report"]["DxItems"]
    if set(d_records) != set(h_records):
        raise ValueError("D/H structured_report DxItem keys do not match")

    roi_entries = list(iter_rois(payload))
    backend_statuses: list[str] = []
    result_sources: dict[str, str] = {}
    selection_summaries: dict[str, dict[str, Any]] = {}

    if not d_records:
        for _, roi in roi_entries:
            roi["selection_history"].append(
                _clee_event(
                    config=config,
                    support=support,
                    dx_pair_id=None,
                    action="inference_skipped",
                    status="skipped",
                    reason="no_diagnostic_pair",
                )
            )

    for DxItem, d_record in d_records.items():
        h_record = h_records[DxItem]
        dx_pair_id = str(d_record["dx_pair_id"])
        if h_record.get("dx_pair_id") != dx_pair_id:
            raise ValueError(f"D/H dx_pair_id mismatch for {DxItem}")
        case_result, unsupported_reason, result_source = _record_support(
            DxItem=DxItem,
            record=h_record,
            support=support,
        )
        result_sources[DxItem] = result_source

        all_query_ids = [
            str(query["query_id"])
            for query in h_record.get("visualAttrQueries", [])
        ]
        eligible_roi_ids: set[str] = set()
        selected_query_ids_by_roi: dict[str, list[str]] = {}
        reference_stains = {str(value) for value in d_record.get("referenceWSI", [])}
        for stain, roi in roi_entries:
            if reference_stains and str(stain["stain_id"]) not in reference_stains:
                continue
            selected_events = [
                event
                for event in _matching_events(roi, dx_pair_id)
                if event["status"] == "selected" and event["selected"] is True
            ]
            query_ids = [
                str(event["query_id"])
                for event in selected_events
                if event.get("query_id")
            ]
            if query_ids:
                eligible_roi_ids.add(roi["roi_id"])
                selected_query_ids_by_roi[roi["roi_id"]] = query_ids

        if unsupported_reason is not None:
            for _, roi in roi_entries:
                roi["selection_history"].append(
                    _clee_event(
                        config=config,
                        support=support,
                        dx_pair_id=dx_pair_id,
                        action="inference_skipped",
                        status="skipped",
                        reason=unsupported_reason,
                        query_ids=all_query_ids,
                    )
                )
            continue

        if not eligible_roi_ids:
            backend_statuses.append("skipped_no_eligible_rois")
            for _, roi in roi_entries:
                roi["selection_history"].append(
                    _clee_event(
                        config=config,
                        support=support,
                        dx_pair_id=dx_pair_id,
                        action="inference_skipped",
                        status="skipped",
                        reason="upstream_visual_filter_rejected",
                        query_ids=all_query_ids,
                    )
                )
            continue

        inference_payload = _inference_payload(
            source_payload=payload,
            eligible_roi_ids=eligible_roi_ids,
            DxItem=DxItem,
            case_result=str(case_result),
        )
        backend_result: BackendResult = run_backend(
            input_payload=inference_payload,
            DxItem=DxItem,
            case_label=str(case_result),
            support=support,
            config=config,
        )
        backend_statuses.append(backend_result.execution_status)
        if backend_result.selection_summary:
            selection_summaries[DxItem] = dict(
                backend_result.selection_summary
            )

        if backend_result.execution_status != "backend_deferred":
            returned_ids = set(backend_result.predictions_by_roi_id)
            if returned_ids != eligible_roi_ids:
                raise ValueError(
                    "CLEE output ROI set does not match submitted eligible ROI set: "
                    f"missing={sorted(eligible_roi_ids - returned_ids)}, "
                    f"unexpected={sorted(returned_ids - eligible_roi_ids)}"
                )

        for _, roi in roi_entries:
            roi_id = roi["roi_id"]
            if roi_id not in eligible_roi_ids:
                roi["selection_history"].append(
                    _clee_event(
                        config=config,
                        support=support,
                        dx_pair_id=dx_pair_id,
                        action="inference_skipped",
                        status="skipped",
                        reason="upstream_visual_filter_rejected",
                        query_ids=all_query_ids,
                    )
                )
                continue
            if backend_result.execution_status == "backend_deferred":
                roi["selection_history"].append(
                    _clee_event(
                        config=config,
                        support=support,
                        dx_pair_id=dx_pair_id,
                        action="inference_skipped",
                        status="skipped",
                        reason="clee_backend_deferred",
                        query_ids=selected_query_ids_by_roi[roi_id],
                    )
                )
                continue

            trace = backend_result.predictions_by_roi_id[roi_id]
            _merge_pseudo_dx_pair(roi, DxItem, trace)
            roi["selection_history"].append(
                _prediction_event(
                    config=config,
                    support=support,
                    dx_pair_id=dx_pair_id,
                    DxItem=DxItem,
                    case_result=str(case_result),
                    trace=trace,
                    selection_detail=(
                        backend_result.selection_by_roi_id.get(roi_id)
                    ),
                    query_ids=selected_query_ids_by_roi[roi_id],
                )
            )

    clee_events = [
        event
        for _, roi in roi_entries
        for event in roi["selection_history"]
        if event["stage"] == "clee"
    ]
    payload.setdefault("reference_versions", {})["clee_inference"] = {
        "backend": config["backend"]["mode"],
        "execution_statuses": sorted(set(backend_statuses)) or ["not_invoked"],
        "checkpoint_config_path": sibling_logical_path(support.config_path),
        "threshold_bundle_path": sibling_logical_path(
            support.threshold_bundle_path
        ),
        "checkpoint_epoch": support.checkpoint_epoch,
        "bundle_id": support.bundle_id,
        "threshold_policy": support.threshold_policy,
        "threshold_resolutions": support.threshold_resolution_records,
        "selection_scope": "case_x_DxItem",
        "zero_candidate_fallback_policy": "terminal_layer_top_k",
        "fallback_top_k": support.fallback_top_k,
        "selection_summaries": selection_summaries,
        "support_policy": (
            "checkpoint_active_label_space"
            if config.get("wlw_supported_dx_items") is None
            else "configured_intersection"
        ),
        "configured_wlw_supported_dx_items": (
            None
            if config.get("wlw_supported_dx_items") is None
            else list(config["wlw_supported_dx_items"])
        ),
        "wlw_supported_dx_items": list(support.effective_DxItem_list),
        "checkpoint_active_dx_items": list(support.active_DxItem_list),
        "effective_dx_items": list(support.effective_DxItem_list),
        "active_dx_result_dict": {
            item: list(results)
            for item, results in support.active_DxResult_dict.items()
        },
        "case_result_sources": result_sources,
        "max_roi_dxitem_inputs_per_forward": int(
            config["max_roi_dxitem_inputs_per_forward"]
        ),
        "evaluated_roi_count": sum(
            event["action"] == "evidence_evaluated" for event in clee_events
        ),
        "selected_roi_count": sum(
            event["status"] == "selected" for event in clee_events
        ),
        "rejected_roi_count": sum(
            event["status"] == "rejected" for event in clee_events
        ),
        "skipped_roi_count": sum(
            event["status"] == "skipped" for event in clee_events
        ),
    }

    artifact = {
        "contract": "I.CLEESelectedROIs",
        "schema_version": "2.0",
        "artifact_id": f"I-{case_id}",
        "case_id": case_id,
        "producer": config["component_version"],
        "payload": payload,
    }
    write_artifact(artifact, args.output, "I.CLEESelectedROIs")


if __name__ == "__main__":
    main()
