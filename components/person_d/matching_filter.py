"""Person D visual attribute matching filter.

The production policy forwards an ROI when either PLIP or CONCH predicts a
label that maps to any informative diagnostic-criteria condition.  Model
agreement, condition polarity, Must gates, and the historical score remain
auditable but do not block forwarding.  The former strict matching behavior is
retained as an explicit compatibility policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


CONDITIONS = (
    "Must_True",
    "Must_False",
    "High_Possibly_True",
    "Low_Possibly_True",
    "Negligible",
    "Not_Mentioned",
)
INFORMATIVE_CONDITIONS = frozenset(
    {"Must_True", "Must_False", "High_Possibly_True", "Low_Possibly_True"}
)
SUPPORTING_CONDITIONS = frozenset({"Must_True", "High_Possibly_True"})
UNVERIFIED_MUST_TRUE_POLICIES = ("skip", "reject")
SELECTION_POLICIES = ("strict_match", "forward_all_informative")
FORWARDED_CONDITIONS = (
    "Must_True",
    "High_Possibly_True",
    "Low_Possibly_True",
    "Must_False",
)


@dataclass(frozen=True)
class MatchingSettings:
    condition_weights: dict[str, float]
    score_threshold: float
    min_evaluated_attributes: int
    unverified_must_true: str
    selection_policy: str

    @classmethod
    def from_config(cls, raw: Mapping[str, Any]) -> "MatchingSettings":
        try:
            weights = {str(key): float(value) for key, value in raw["condition_weights"].items()}
            settings = cls(
                condition_weights=weights,
                score_threshold=float(raw["score_threshold"]),
                min_evaluated_attributes=int(raw["min_evaluated_attributes"]),
                unverified_must_true=str(raw["unverified_must_true"]),
                selection_policy=str(raw.get("selection_policy", "strict_match")),
            )
        except KeyError as exc:
            raise ValueError(f"Person D matching config is missing {exc.args[0]!r}") from exc
        if set(weights) != INFORMATIVE_CONDITIONS:
            raise ValueError(
                "condition_weights must define exactly "
                f"{sorted(INFORMATIVE_CONDITIONS)}"
            )
        if any(not -1.0 <= value <= 1.0 for value in weights.values()):
            raise ValueError("condition_weights must stay within [-1, 1]")
        if not -1.0 <= settings.score_threshold <= 1.0:
            raise ValueError("score_threshold must stay within [-1, 1]")
        if settings.min_evaluated_attributes < 1:
            raise ValueError("min_evaluated_attributes must be at least 1")
        if settings.unverified_must_true not in UNVERIFIED_MUST_TRUE_POLICIES:
            raise ValueError(
                f"unverified_must_true must be one of {UNVERIFIED_MUST_TRUE_POLICIES}"
            )
        if settings.selection_policy not in SELECTION_POLICIES:
            raise ValueError(f"selection_policy must be one of {SELECTION_POLICIES}")
        return settings


@dataclass(frozen=True)
class LabelMap:
    """Extraction-label to criteria-option aliases for one criteria version."""

    criteria_version: str
    global_aliases: dict[str, list[str]]
    attribute_aliases: dict[str, dict[str, list[str]]]

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "LabelMap":
        return cls(
            criteria_version=str(document["criteria_version"]),
            global_aliases={
                str(label): list(options)
                for label, options in document["global_aliases"].items()
            },
            attribute_aliases={
                str(key): {str(label): list(options) for label, options in table.items()}
                for key, table in document["attribute_aliases"].items()
            },
        )

    def resolve(self, attribute_key: str, label: str, options: list[str]) -> list[str]:
        """Return the criteria options for one label: alias, exact, then case-insensitive."""

        for table in (self.attribute_aliases.get(attribute_key, {}), self.global_aliases):
            if label in table:
                return [option for option in table[label] if option in options]
        if label in options:
            return [label]
        return [option for option in options if option.lower() == label.lower()]


def evaluate_attribute(
    labels: list[str],
    attr_spec: Mapping[str, Any],
    attribute_key: str,
    label_map: LabelMap,
) -> dict[str, Any]:
    result: dict[str, Any] = {"labels": list(labels)}
    if not labels:
        result["status"] = "models_disagree"
        return result

    options: list[str] = []
    conditions: list[str] = []
    unmapped: list[str] = []
    ambiguous: list[str] = []
    for label in labels:
        mapped = label_map.resolve(attribute_key, label, list(attr_spec["options"]))
        if not mapped:
            unmapped.append(label)
            continue
        label_conditions = {
            attr_spec["conditions"].get(option, "Not_Mentioned") for option in mapped
        }
        if len(label_conditions) > 1:
            ambiguous.append(label)
            continue
        options.extend(mapped)
        conditions.append(label_conditions.pop())

    result.update(options=options, conditions=conditions)
    if unmapped:
        result["unmapped_labels"] = unmapped
    if ambiguous:
        result["ambiguous_labels"] = ambiguous
    if not conditions:
        result["status"] = "ambiguous_label" if ambiguous else "unmapped_label"
    elif not INFORMATIVE_CONDITIONS & set(conditions):
        result["status"] = "uninformative"
    else:
        result["status"] = "evaluated"
    return result


def _prediction_labels(info: Any) -> list[str]:
    """Normalize one model's serialized prediction to a label list."""

    if not isinstance(info, Mapping):
        return []
    prediction = info.get("prediction")
    if prediction is None:
        return []
    if isinstance(prediction, list):
        return [str(label) for label in prediction if label is not None]
    return [str(prediction)]


def _mean_condition_score(
    evaluated: Mapping[str, Mapping[str, Any]], settings: MatchingSettings
) -> float:
    attribute_scores = []
    for result in evaluated.values():
        weights = [
            settings.condition_weights[condition]
            for condition in result["conditions"]
            if condition in settings.condition_weights
        ]
        attribute_scores.append(sum(weights) / len(weights))
    return round(sum(attribute_scores) / len(attribute_scores), 6) if attribute_scores else 0.0


def match_query(
    extracted: Mapping[str, Mapping[str, list[str]]],
    criteria: Mapping[str, Any],
    label_map: LabelMap,
    settings: MatchingSettings,
    vocabulary: Mapping[str, Mapping[str, list[str]]],
    model_predictions: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Match one ROI's consistent labels against one diagnosticCriteria.

    ``vocabulary`` is the extraction vocabulary (category -> attribute -> labels); it decides
    which criteria options the backend can reach at all.
    """

    if criteria["version"] != label_map.criteria_version:
        raise ValueError(
            f"diagnosticCriteria version {criteria['version']!r} does not match "
            f"label map criteria_version {label_map.criteria_version!r}"
        )

    attributes: dict[str, dict[str, Any]] = {}
    attribute_specs: dict[str, Mapping[str, Any]] = {}
    for category, category_spec in criteria["visualAttrs"].items():
        for attribute, attr_spec in category_spec.items():
            key = f"{category}.{attribute}"
            attribute_specs[key] = attr_spec
            labels = extracted.get(category, {}).get(attribute)
            if labels is None:
                result: dict[str, Any] = {"status": "not_extracted"}
            else:
                result = evaluate_attribute(labels, attr_spec, key, label_map)
            result["requires_must_true"] = "Must_True" in attr_spec["conditions"].values()
            options = list(attr_spec["options"])
            reachable = {
                option
                for label in vocabulary.get(category, {}).get(attribute, [])
                for option in label_map.resolve(key, label, options)
            }
            unreachable_must = {
                condition: [
                    option
                    for option in options
                    if attr_spec["conditions"].get(option) == condition and option not in reachable
                ]
                for condition in ("Must_True", "Must_False")
            }
            unreachable_must = {condition: found for condition, found in unreachable_must.items() if found}
            if unreachable_must:
                result["unreachable_must"] = unreachable_must
            attributes[key] = result

    model_attributes: dict[str, dict[str, dict[str, Any]]] = {}
    if model_predictions:
        for model_name, prediction in model_predictions.items():
            per_model: dict[str, dict[str, Any]] = {}
            for category, category_spec in criteria["visualAttrs"].items():
                for attribute, attr_spec in category_spec.items():
                    key = f"{category}.{attribute}"
                    info = prediction.get(category, {}).get(attribute)
                    labels = _prediction_labels(info)
                    if labels:
                        per_model[key] = evaluate_attribute(
                            labels, attr_spec, key, label_map
                        )
                    else:
                        per_model[key] = {"labels": [], "status": "no_prediction"}
            model_attributes[str(model_name)] = per_model

    evaluated = {key: result for key, result in attributes.items() if result["status"] == "evaluated"}
    must_false_failed = sorted(
        key for key, result in evaluated.items() if "Must_False" in result["conditions"]
    )
    # A miss only counts when every Must_True option of the attribute could have been predicted.
    must_true_failed = sorted(
        key
        for key, result in evaluated.items()
        if result["requires_must_true"]
        and "Must_True" not in result["conditions"]
        and "Must_True" not in result.get("unreachable_must", {})
    )
    must_true_unverified = sorted(
        key
        for key, result in attributes.items()
        if result["requires_must_true"] and key not in evaluated
    )
    must_unverifiable = sorted(key for key, result in attributes.items() if "unreachable_must" in result)

    score = _mean_condition_score(evaluated, settings)

    unverified = sorted(set(must_true_unverified) | set(must_unverifiable))
    reject_unverified = settings.unverified_must_true == "reject" and bool(unverified)
    must_false_passed = not must_false_failed and not any(
        "Must_False" in attributes[key]["unreachable_must"] for key in must_unverifiable
    )
    must_true_passed = (
        not must_true_failed
        and not must_true_unverified
        and not any("Must_True" in attributes[key]["unreachable_must"] for key in must_unverifiable)
    )
    failed = sorted(
        set(must_false_failed)
        | set(must_true_failed)
        | (set(unverified) if reject_unverified else set())
    )
    matched = sorted(
        key
        for key, result in evaluated.items()
        if SUPPORTING_CONDITIONS & set(result["conditions"])
    )

    forwarding_sources = model_attributes or {"CONSENSUS": attributes}
    forwarded_attributes = sorted(
        {
            key
            for per_model in forwarding_sources.values()
            for key, result in per_model.items()
            if result["status"] == "evaluated"
        }
    )
    condition_hits: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in FORWARDED_CONDITIONS
    }
    model_scores: dict[str, float] = {}
    for model_name, per_model in forwarding_sources.items():
        model_evaluated = {
            key: result
            for key, result in per_model.items()
            if result["status"] == "evaluated"
        }
        model_scores[model_name] = _mean_condition_score(model_evaluated, settings)
        for key, result in model_evaluated.items():
            spec = attribute_specs[key]
            for condition in FORWARDED_CONDITIONS:
                if condition not in result["conditions"]:
                    continue
                condition_hits[condition].append(
                    {
                        "model": model_name,
                        "attribute": key,
                        "labels": list(result["labels"]),
                        "options": [
                            option
                            for option in result["options"]
                            if spec["conditions"].get(option, "Not_Mentioned") == condition
                        ],
                    }
                )

    if settings.selection_policy == "forward_all_informative":
        if len(forwarded_attributes) < settings.min_evaluated_attributes:
            status, reason = "skipped", "insufficient_visual_evidence"
        else:
            status, reason = "selected", "informative_visual_attributes_forwarded"
    else:
        # In strict compatibility mode, an unverified Must condition is
        # checked before the threshold and never ends as selected.
        if len(evaluated) < settings.min_evaluated_attributes:
            status, reason = "skipped", "insufficient_visual_evidence"
        elif must_false_failed or must_true_failed or reject_unverified:
            status, reason = "rejected", "must_condition_failed"
        elif unverified:
            status, reason = "skipped", "insufficient_visual_evidence"
        elif score < settings.score_threshold:
            status, reason = "rejected", "score_below_threshold"
        else:
            status, reason = "selected", "visual_attributes_match"

    return {
        "status": status,
        "reason": reason,
        "score": score,
        "must_true_passed": must_true_passed,
        "must_false_passed": must_false_passed,
        "matched_attributes": matched,
        "failed_attributes": failed,
        "selection_policy": settings.selection_policy,
        "evaluated_count": (
            len(forwarded_attributes)
            if settings.selection_policy == "forward_all_informative"
            else len(evaluated)
        ),
        "consensus_evaluated_count": len(evaluated),
        "consensus_score": score,
        "model_scores": model_scores,
        "forwarded_attributes": forwarded_attributes,
        "condition_hits": condition_hits,
        "model_attributes": model_attributes,
        "must_true_unverified": must_true_unverified,
        "must_unverifiable": must_unverifiable,
        "attributes": attributes,
    }
