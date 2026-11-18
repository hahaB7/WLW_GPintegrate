from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any

from components.person_a.learnable_query_generator import (
    LearnableQueryGenerationError,
)
from components.person_a.reference_data import (
    load_histologic_mapping,
    load_visual_template,
)
from contracts.metadata import dx_items, ensure_same_case
from contracts.runtime import cli_parser, load_config, load_inputs, write_artifact


CONDITION_LABELS = {
    "Must_True",
    "Must_False",
    "High_Possibly_True",
    "Low_Possibly_True",
    "Negligible",
    "Not_Mentioned",
}


def _index_chunks(
    dx_payload: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Validate F linkage and return chunks ordered by retrieval rank."""

    dx_pair_ids = [
        str(record["dx_pair_id"]) for record in dx_items(dx_payload).values()
    ]
    if len(dx_pair_ids) != len(set(dx_pair_ids)):
        raise ValueError("D contains duplicate dx_pair_id values")
    known_dx_pairs = set(dx_pair_ids)

    by_dx: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_chunk_ids: set[str] = set()
    seen_ranks: dict[str, set[int]] = defaultdict(set)
    foreign_dx_pairs: set[str] = set()

    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        dx_pair_id = str(chunk["dx_pair_id"])
        retrieval_rank = int(chunk["retrieval_rank"])
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"F contains duplicate chunk_id {chunk_id!r}")
        seen_chunk_ids.add(chunk_id)
        if dx_pair_id not in known_dx_pairs:
            foreign_dx_pairs.add(dx_pair_id)
            continue
        if retrieval_rank in seen_ranks[dx_pair_id]:
            raise ValueError(
                f"F contains duplicate retrieval_rank {retrieval_rank} for "
                f"DxPair {dx_pair_id!r}"
            )
        seen_ranks[dx_pair_id].add(retrieval_rank)
        by_dx[dx_pair_id].append(chunk)

    if foreign_dx_pairs:
        raise ValueError(
            "F contains chunks for dx_pair_id values absent from D: "
            f"{sorted(foreign_dx_pairs)}"
        )

    for dx_pair_id in by_dx:
        by_dx[dx_pair_id].sort(key=lambda chunk: int(chunk["retrieval_rank"]))
    return by_dx


def _condition_schema(visual_attrs: dict[str, Any]) -> dict[str, Any]:
    """Return the model-output shape without any subtype-level values."""

    return {
        category: {
            feature: {option: "Not_Mentioned" for option in definition["options"]}
            for feature, definition in features.items()
        }
        for category, features in visual_attrs.items()
    }


def _build_learned_criteria(
    template: dict[str, Any],
    mapping_entry: dict[str, Any],
    generated: dict[str, Any],
) -> dict[str, Any]:
    """Build diagnostic criteria solely from a template and generated values."""

    visual_attrs = template.get("visualAttrs")
    if not isinstance(visual_attrs, dict):
        raise ValueError("Visual attribute template has no visualAttrs object")
    if set(generated) != set(visual_attrs):
        raise ValueError(
            "Learned Attribute groups do not match the inference template; "
            f"expected={sorted(visual_attrs)}, actual={sorted(generated)}"
        )

    criteria_visual_attrs: dict[str, dict[str, Any]] = {}
    option_types: list[str] = []
    for category, features in visual_attrs.items():
        criteria_visual_attrs[category] = {}
        generated_features = generated.get(category)
        if not isinstance(generated_features, dict) or set(generated_features) != set(
            features
        ):
            raise ValueError(
                f"Generated Attribute category {category!r} has different features"
            )
        for feature, definition in features.items():
            if not isinstance(definition, dict):
                raise ValueError(
                    f"Template visual attribute {category}.{feature} is not an object"
                )
            options = definition.get("options")
            option_type = definition.get("type")
            if not isinstance(options, list) or not isinstance(option_type, str):
                raise ValueError(
                    f"Template visual attribute {category}.{feature} lacks options/type"
                )
            generated_conditions = generated_features[feature]
            if not isinstance(generated_conditions, dict):
                raise ValueError(
                    f"Generated {category}.{feature} must map every option "
                    "to one condition"
                )
            if set(generated_conditions) != set(options):
                raise ValueError(
                    f"Generated {category}.{feature} options do not match G; "
                    f"expected={sorted(options)}, "
                    f"actual={sorted(generated_conditions)}"
                )
            illegal_conditions = sorted(
                {
                    repr(value)
                    for value in generated_conditions.values()
                    if not isinstance(value, str) or value not in CONDITION_LABELS
                }
            )
            if illegal_conditions:
                raise ValueError(
                    f"Generated {category}.{feature} has illegal conditions: "
                    f"{illegal_conditions}"
                )
            if option_type not in option_types:
                option_types.append(option_type)
            criteria_visual_attrs[category][feature] = {
                "options": deepcopy(options),
                "type": option_type,
                "conditions": {
                    option: generated_conditions[option] for option in options
                },
            }
    return {
        "version": template["version"],
        "type": "diagnosticCriteria",
        "includeReferences": [
            "visual_attribute_template",
            "generated_conditions",
            "histologic_type_mapping",
        ],
        "typeLevel_idx": mapping_entry["typeLevel_idx"],
        "DxItem": "Histologic_Type",
        "DxResult": mapping_entry["DxResult"],
        "attrConditions": deepcopy(template["conditionLabels"]),
        "optionTypes": option_types,
        "visualAttrs": criteria_visual_attrs,
    }


def _learnable_generator(
    backend_config: dict[str, Any],
    condition_schema: dict[str, Any],
    template_provenance: dict[str, Any],
) -> Any:
    from components.person_a.learnable_query_generator import (
        LearnableSoftPromptGenerator,
    )

    return LearnableSoftPromptGenerator(
        backend_config,
        condition_schema=condition_schema,
        template_provenance=template_provenance,
    )


def build_example_artifact(
    d_artifact: dict[str, Any],
    f_artifact: dict[str, Any],
    producer: str,
) -> dict[str, Any]:
    """Emit deterministic empty queries for the public contract smoke test."""

    case_id = ensure_same_case(d_artifact, f_artifact)
    payload = deepcopy(d_artifact["payload"])
    for record in payload["case_list"][0]["structured_report"]["DxItems"].values():
        record["visualAttrQueries"] = []
    payload.setdefault("reference_versions", {})["example_query_stub"] = {
        "implemented": False,
        "purpose": "contract smoke test",
    }
    return {
        "contract": "G.VisualAttributeQueries",
        "schema_version": "2.0",
        "artifact_id": f"G-example-{case_id}",
        "case_id": case_id,
        "producer": producer,
        "payload": payload,
    }


def generate_artifact(
    dx_artifact: dict[str, Any],
    chunks_artifact: dict[str, Any],
    config: dict[str, Any],
    generator: Any | None = None,
) -> dict[str, Any]:
    case_id = ensure_same_case(dx_artifact, chunks_artifact)
    payload = deepcopy(dx_artifact["payload"])
    integrity_mode = config.get("asset_integrity", "enforce")

    mapping, mapping_provenance = load_histologic_mapping(
        config["histologic_mapping_path"],
        integrity_mode=integrity_mode,
    )
    visual_template, template_provenance = load_visual_template(
        config["visual_attribute_template_path"],
        integrity_mode=integrity_mode,
    )
    if set(visual_template["conditionLabels"]) != CONDITION_LABELS:
        raise ValueError(
            "Visual attribute template conditionLabels do not match the runtime "
            "condition vocabulary"
        )
    condition_schema = _condition_schema(visual_template["visualAttrs"])
    payload["candidateReference"] = {
        "visualAttrs": deepcopy(visual_template["visualAttrs"])
    }
    payload.setdefault("reference_versions", {})[
        "visual_attribute_template"
    ] = template_provenance
    payload["reference_versions"]["histologic_type_mapping"] = mapping_provenance

    backend_config = deepcopy(config.get("backend") or {})
    backend_config["asset_integrity"] = integrity_mode
    backend_mode = str(backend_config.get("mode", ""))
    if backend_mode != "learnable_soft_prompt":
        raise ValueError(
            "query generation backend.mode must be learnable_soft_prompt; "
            "inference has no condition-reference fallback"
        )

    by_dx = _index_chunks(
        dx_artifact["payload"],
        chunks_artifact["payload"]["chunks"],
    )

    query_index = 1
    learnable_provenance: dict[str, Any] | None = None
    learnable_generation_attempts: list[dict[str, Any]] = []
    selected_chunks_by_dx_pair: dict[str, list[dict[str, Any]]] = {}
    for dx_item, record in dx_items(payload).items():
        record["visualAttrQueries"] = []
        if dx_item != "Histologic_Type":
            continue
        dx_result = record["DxResultCls"]
        if not isinstance(dx_result, str):
            raise ValueError("Histologic_Type DxResultCls must be a single string")
        mapping_entry = mapping.get(dx_result)
        mapped_result = mapping_entry["DxResult"] if mapping_entry else None
        criteria: dict[str, Any] | None = None
        chunks = by_dx[record["dx_pair_id"]]
        used_chunks = chunks
        if mapping_entry is not None:
            required_chunk_count = int(
                backend_config.get("required_chunk_count", 3)
            )
            if len(chunks) < required_chunk_count:
                raise ValueError(
                    f"DxPair {record['dx_pair_id']!r} has {len(chunks)} F chunks; "
                    f"the trained soft prompt requires {required_chunk_count}"
                )
            used_chunks = chunks[:required_chunk_count]
            selected_chunks_by_dx_pair[record["dx_pair_id"]] = [
                {
                    "chunk_id": chunk["chunk_id"],
                    "retrieval_rank": int(chunk["retrieval_rank"]),
                }
                for chunk in used_chunks
            ]
            if generator is None:
                generator = _learnable_generator(
                    backend_config,
                    condition_schema,
                    template_provenance,
                )
            dx_text = record.get("DxResultRawTxt") or record["DxResultTxt"]
            retry_with_canonical_dx = bool(
                backend_config.get("retry_with_canonical_dx", False)
            )
            try:
                generated = generator.generate(dx_text, used_chunks)
            except LearnableQueryGenerationError as first_error:
                first_diagnostics = deepcopy(
                    getattr(generator, "last_generation_diagnostics", {})
                )
                learnable_generation_attempts.append(
                    {
                        "dx_pair_id": record["dx_pair_id"],
                        "input_strategy": "raw_dx_text",
                        "status": "rejected",
                        **first_diagnostics,
                        "error": str(first_error).split(
                            "; response context=", 1
                        )[0],
                    }
                )
                canonical_dx_text = str(mapped_result or "").strip()
                if (
                    not retry_with_canonical_dx
                    or not canonical_dx_text
                    or canonical_dx_text == str(dx_text).strip()
                ):
                    raise
                generated = generator.generate(canonical_dx_text, used_chunks)
                learnable_generation_attempts.append(
                    {
                        "dx_pair_id": record["dx_pair_id"],
                        "input_strategy": "canonical_mapped_dx",
                        "status": "accepted",
                        **deepcopy(
                            getattr(
                                generator,
                                "last_generation_diagnostics",
                                {},
                            )
                        ),
                    }
                )
            else:
                learnable_generation_attempts.append(
                    {
                        "dx_pair_id": record["dx_pair_id"],
                        "input_strategy": "raw_dx_text",
                        "status": "accepted",
                        **deepcopy(
                            getattr(
                                generator,
                                "last_generation_diagnostics",
                                {},
                            )
                        ),
                    }
                )
            criteria = _build_learned_criteria(
                visual_template,
                mapping_entry,
                generated,
            )
            learnable_provenance = deepcopy(generator.provenance)
            learnable_provenance["condition_source"] = "model_generation_only"
        evidence = " ".join(chunk["text"] for chunk in used_chunks)
        status = "mapped" if criteria is not None else "unmapped"
        mapping_source = (
            f"{mapping_provenance['source_name']}#sha256="
            f"{mapping_provenance['sha256']}"
            f";{template_provenance['source_name']}#sha256="
            f"{template_provenance['sha256']}"
        )
        if learnable_provenance is not None and criteria is not None:
            mapping_source += (
                ";soft_prompt#sha256="
                f"{learnable_provenance['checkpoint_sha256']}"
            )
        record["visualAttrQueries"].append(
            {
                "query_id": f"{case_id}-query-{query_index:03d}",
                "dx_pair_id": record["dx_pair_id"],
                "text": evidence or f"Evaluate visual criteria for {dx_result}.",
                "criteria_status": status,
                "diagnosticCriteria": deepcopy(criteria),
                "mapping_source": mapping_source,
                "chunk_ids": [chunk["chunk_id"] for chunk in used_chunks],
            }
        )
        query_index += 1

    if learnable_provenance is not None:
        learnable_provenance["chunk_selection_rule"] = (
            "ascending_retrieval_rank_top_n"
        )
        learnable_provenance["selected_chunks_by_dx_pair"] = (
            selected_chunks_by_dx_pair
        )
        learnable_provenance["canonical_dx_retry_enabled"] = bool(
            backend_config.get("retry_with_canonical_dx", False)
        )
        learnable_provenance["generation_attempts"] = (
            learnable_generation_attempts
        )
        payload["reference_versions"][
            "learnable_query_generation"
        ] = learnable_provenance

    return {
        "contract": "G.VisualAttributeQueries",
        "schema_version": "2.0",
        "artifact_id": f"G-{case_id}",
        "case_id": case_id,
        "producer": config["component_version"],
        "payload": payload,
    }


def main() -> None:
    args = cli_parser("甲: generate G visual attribute queries from F chunks").parse_args()
    inputs = load_inputs(args.input, ["D.DxPairs", "F.Chunks"])
    config = load_config(args.config)
    if config.get("mode") == "example":
        artifact = build_example_artifact(
            inputs["D.DxPairs"],
            inputs["F.Chunks"],
            config["component_version"],
        )
    else:
        artifact = generate_artifact(
            inputs["D.DxPairs"],
            inputs["F.Chunks"],
            config,
        )
    write_artifact(artifact, args.output, "G.VisualAttributeQueries")


if __name__ == "__main__":
    main()
