"""Self-contained, inference-only CLEE runtime for the WLW Person E component."""

from __future__ import annotations

from contextlib import ExitStack, nullcontext
from copy import copy, deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from PIL import Image
import torch
import torch.nn.functional as F

from components.person_e.embedding_catalog import (
    load_embedding_bank as load_catalog_embedding_bank,
    normalize_embedding_key,
)
from components.person_e.model.contracts import Case, ROI
from contracts.paths import resolve_person_reference_path, resolve_run_path


FINAL_RESULT_KEY = "finalResult"
UNABLE_TO_DETERMINE = "unable_to_determine"
HISTOLOGIC_TYPE_SEVERITY = (
    "Normal(N)",
    "Pathological Benign(PB)",
    "Usual Ductal Hyperplasia(UDH)",
    "Flat Epithelial Atypia(FEA)",
    "Atypical Ductal Hyperplasia(ADH)",
    "Ductal Carcinoma in Situ(DCIS)",
    "Invasive Carcinoma(IC)",
)


@dataclass(frozen=True)
class NativeArtifacts:
    checkpoint_dir: Path
    config_path: Path
    model_checkpoint_path: Path
    threshold_bundle_path: Path
    checkpoint_epoch: int


@dataclass(frozen=True)
class _InputRecord:
    roi_id: str
    roi: ROI


@dataclass(frozen=True)
class NativeInferenceResult:
    predictions_by_roi_id: dict[str, dict[str, Any]]
    selection_by_roi_id: dict[str, dict[str, Any]]
    selection_summary: dict[str, Any]


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _path(raw: Any, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Native CLEE requires a nonempty {name}")
    return resolve_person_reference_path("person_e", raw)


def resolve_native_artifacts(config: Mapping[str, Any]) -> NativeArtifacts:
    checkpoint_dir = _path(config.get("checkpoint_dir"), "checkpoint_dir")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"CLEE checkpoint directory is missing: {checkpoint_dir}")

    config_path = checkpoint_dir / "config.json"
    manifest_path = checkpoint_dir / "[checkpoint]best_model.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"CLEE checkpoint config is missing: {config_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CLEE checkpoint manifest is missing: {manifest_path}")

    manifest = _json_object(manifest_path)
    metadata = manifest.get("checkpoint_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("epoch_idx") is None:
        raise ValueError(f"checkpoint_metadata.epoch_idx is missing in {manifest_path}")
    epoch = int(metadata["epoch_idx"])
    model_filename = manifest.get("model_filename")
    if not isinstance(model_filename, str) or not model_filename:
        raise ValueError(f"model_filename is missing in {manifest_path}")
    model_checkpoint_path = checkpoint_dir / model_filename
    if not model_checkpoint_path.is_file():
        raise FileNotFoundError(f"CLEE prefix checkpoint is missing: {model_checkpoint_path}")

    configured_threshold = config.get("threshold_bundle_path")
    if configured_threshold:
        threshold_path = _path(configured_threshold, "threshold_bundle_path")
    else:
        prefix = f"threshold_bundle[{epoch:03d}]_"
        matches = sorted(
            path
            for path in checkpoint_dir.iterdir()
            if path.name.startswith(prefix) and path.suffix == ".json"
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one threshold bundle for checkpoint epoch {epoch}; "
                f"found {len(matches)}"
            )
        threshold_path = matches[0]
    if not threshold_path.is_file():
        raise FileNotFoundError(f"CLEE threshold bundle is missing: {threshold_path}")

    threshold = _json_object(threshold_path)
    if int(threshold.get("epoch_idx", -1)) != epoch:
        raise ValueError(
            f"CLEE checkpoint/threshold epoch mismatch: {epoch} != "
            f"{threshold.get('epoch_idx')!r}"
        )
    return NativeArtifacts(
        checkpoint_dir=checkpoint_dir,
        config_path=config_path,
        model_checkpoint_path=model_checkpoint_path,
        threshold_bundle_path=threshold_path,
        checkpoint_epoch=epoch,
    )


def _native_options(config: Mapping[str, Any]) -> Mapping[str, Any]:
    backend = config.get("backend")
    if not isinstance(backend, Mapping):
        raise ValueError("CLEE config requires a backend object")
    options = backend.get("model")
    if not isinstance(options, Mapping):
        raise ValueError("backend.mode='native' requires backend.model")
    return options


def _embedding_path(
    checkpoint_config: Mapping[str, Any], options: Mapping[str, Any], weight_path: Path
) -> Path:
    explicit = options.get("embedding_path")
    if explicit:
        result = _path(explicit, "backend.model.embedding_path")
    else:
        checkpoint_value = checkpoint_config.get("embedding_path")
        if not isinstance(checkpoint_value, str) or not checkpoint_value:
            raise ValueError("Checkpoint config has no embedding_path")
        result = weight_path / Path(checkpoint_value).name
    if not result.is_dir():
        raise FileNotFoundError(
            f"CLEE embedding catalog directory is missing: {result}"
        )
    return result


def _validate_checkpoint_metadata(
    checkpoint_config: Mapping[str, Any],
    threshold_bundle: Mapping[str, Any],
    embedding_path: Path,
) -> None:
    if checkpoint_config.get("model_variant", "CLEE") != "CLEE":
        raise ValueError("Native inference requires checkpoint model_variant='CLEE'")
    if normalize_embedding_key(threshold_bundle.get("embedding_key")) != (
        normalize_embedding_key(checkpoint_config.get("embedding_key"))
    ):
        raise ValueError("CLEE checkpoint/threshold embedding_key mismatch")
    threshold_embedding = threshold_bundle.get("embedding_path")
    if not isinstance(threshold_embedding, str) or (
        Path(threshold_embedding).name != embedding_path.name
    ):
        raise ValueError("CLEE checkpoint/threshold embedding catalog mismatch")


def _dtype(name: Any) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported backend.model.dtype: {name!r}")
    return mapping[str(name)]


def _build_model_config(
    checkpoint_config: Mapping[str, Any],
    options: Mapping[str, Any],
    weight_path: Path,
    embedding_path: Path,
) -> SimpleNamespace:
    values = dict(checkpoint_config)
    if values.get("model_name") != "medgemma-1.5-4b-it":
        raise ValueError(
            "The bundled inference-only modelBuilder supports checkpoint "
            "model_name='medgemma-1.5-4b-it'"
        )
    values.update(
        {
            "weight_path": str(weight_path),
            "embedding_path": str(embedding_path),
            "device": str(options.get("device", "cuda")),
            "torch_dtype": _dtype(options.get("dtype", "bfloat16")),
            "attn_implementation": str(options.get("attn_implementation", "sdpa")),
            "pp_num_gpus": options.get("pp_num_gpus"),
            "deep_prefix_debug_log": bool(options.get("deep_prefix_debug_log", False)),
            "deep_prefix_debug_log_freq": int(
                options.get("deep_prefix_debug_log_freq", 50)
            ),
        }
    )
    required = (
        "DxItem_list",
        "active_DxItem_list",
        "active_DxResult_dict",
        "prefix_len",
        "sep_str",
        "boc_str",
        "input_img",
        "input_attr",
        "input_loc",
        "level_key",
        "embedding_key",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"CLEE checkpoint config is missing: {', '.join(missing)}")
    if values["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Native CLEE requested CUDA, but PyTorch cannot access a CUDA device")
    return SimpleNamespace(**values)


def _mpp_pair(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        pair = (float(value), float(value))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        pair = (float(value[0]), float(value[1]))
    else:
        raise ValueError(f"Invalid {name}: {value!r}")
    if any(not math.isfinite(part) or part <= 0 for part in pair):
        raise ValueError(f"{name} must contain positive finite values: {value!r}")
    return pair


def _resolve_data_path(
    raw: Any,
    image_root: Path | None,
    name: str,
    *,
    allow_external_absolute: bool = False,
) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Missing {name}")
    path = Path(raw).expanduser()
    if path.is_absolute() and allow_external_absolute:
        return path.resolve()
    if not path.is_absolute():
        if image_root is None:
            raise ValueError(f"Relative {name} requires image_root: {raw}")
        path = image_root / path
    return resolve_run_path(path)


def _read_wsi_roi(slide: Any, roi: Mapping[str, Any]) -> Image.Image:
    level0 = roi["level0_info"]
    x, y, width, height = (int(value) for value in level0["xywh"])
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"ROI {roi['roi_id']} has invalid level0_info.xywh")
    if x + width > slide.dimensions[0] or y + height > slide.dimensions[1]:
        raise ValueError(f"ROI {roi['roi_id']} exceeds the WSI dimensions")

    level0_mpp = _mpp_pair(level0["mpp"], "level0_info.mpp")
    target_pair = _mpp_pair(roi["main_info"]["mpp"], "main_info.mpp")
    target_mpp = sum(target_pair) / 2.0
    desired = min(target_mpp / level0_mpp[0], target_mpp / level0_mpp[1])
    level = slide.get_best_level_for_downsample(max(1.0, desired))
    downsample = float(slide.level_downsamples[level])
    source_size = (math.ceil(width / downsample), math.ceil(height / downsample))
    rgba = slide.read_region((x, y), level, source_size)
    try:
        image = rgba.convert("RGB")
    finally:
        rgba.close()
    target_size = (
        max(1, round(width * level0_mpp[0] / target_mpp)),
        max(1, round(height * level0_mpp[1] / target_mpp)),
    )
    if image.size != target_size:
        resized = image.resize(target_size, Image.Resampling.BICUBIC)
        image.close()
        image = resized
    return image


def _roi_image(
    stack: ExitStack,
    slides: dict[Path, Any],
    stain: Mapping[str, Any],
    roi: Mapping[str, Any],
    image_root: Path | None,
) -> Image.Image:
    roi_path = roi["main_info"].get("roi_path")
    if roi_path:
        path = _resolve_data_path(roi_path, image_root, "main_info.roi_path")
        if not path.is_file():
            raise FileNotFoundError(f"ROI image is missing: {path}")
        with Image.open(path) as source:
            return source.convert("RGB")

    wsi_path = _resolve_data_path(
        stain.get("filepath"),
        image_root,
        "stain.filepath",
        allow_external_absolute=True,
    )
    if not wsi_path.is_file():
        raise FileNotFoundError(
            f"WSI is missing for stain {stain.get('stain_id')}: {wsi_path}"
        )
    if wsi_path not in slides:
        import openslide

        slides[wsi_path] = stack.enter_context(openslide.OpenSlide(str(wsi_path)))
    return _read_wsi_roi(slides[wsi_path], roi)


def _case_records(
    payload: Mapping[str, Any],
    DxItem: str,
    case_label: str,
    cfg: SimpleNamespace,
    embedding_bank: Mapping[str, Mapping[str, torch.Tensor]],
    image_root: Path | None,
) -> tuple[Case, list[_InputRecord], ExitStack]:
    cases = payload.get("case_list")
    if not isinstance(cases, list) or len(cases) != 1:
        raise ValueError("Native CLEE expects exactly one case")
    case_data = cases[0]
    stack = ExitStack()
    slides: dict[Path, Any] = {}
    records: list[_InputRecord] = []
    try:
        for block in case_data["tissue_blocks"]:
            for stain in block["stains"]:
                for raw_roi in stain["roi_list"]:
                    image = (
                        _roi_image(stack, slides, stain, raw_roi, image_root)
                        if bool(cfg.input_img)
                        else None
                    )
                    if image is not None:
                        stack.callback(image.close)
                    mpp = (
                        sum(_mpp_pair(raw_roi[cfg.level_key]["mpp"], "ROI mpp")) / 2.0
                        if bool(cfg.input_loc)
                        else None
                    )
                    cxcywh = (
                        tuple(float(value) for value in raw_roi[cfg.level_key]["cxcywh"])
                        if bool(cfg.input_loc)
                        else None
                    )
                    visual_attrs = raw_roi.get("visualAttrs") if bool(cfg.input_attr) else None
                    roi = ROI(
                        global_idx=int(raw_roi["global_idx"]),
                        image=image,
                        mpp=mpp,
                        cxcywh=cxcywh,
                        visualAttrs=visual_attrs,
                        roi_wh=tuple(int(value) for value in raw_roi[cfg.level_key]["roi_wh"]),
                        DxItem=DxItem,
                        DxResult=case_label,
                        DxResult_embeddings=embedding_bank[DxItem][case_label],
                    )
                    records.append(_InputRecord(roi_id=str(raw_roi["roi_id"]), roi=roi))

        if len({record.roi_id for record in records}) != len(records):
            raise ValueError("Native CLEE input contains duplicate roi_id values")
        if len({record.roi.global_idx for record in records}) != len(records):
            raise ValueError("Native CLEE input contains duplicate ROI global_idx values")

        if len(records) > 1:
            seed_value = (
                f"{int(getattr(cfg, 'roi_shuffle_seed', 42))}:"
                f"inference:{case_data['sample_idx']}"
            ).encode("utf-8")
            seed = int.from_bytes(sha256(seed_value).digest()[:8], byteorder="big")
            shuffled = list(records)
            random.Random(seed).shuffle(shuffled)
            if all(left is right for left, right in zip(shuffled, records)):
                shuffled = shuffled[1:] + shuffled[:1]
            records = shuffled

        patient = case_data.get("patient_info", {})
        case = Case(
            global_idx=case_data["sample_idx"],
            case_id=case_data["case_id"],
            pid=patient.get("pid"),
            roi_ExistCounts=case_data.get("roi_ExistCounts"),
            rois=[record.roi for record in records],
            DxPair_dict={
                DxItem: {
                    "DxResult": case_label,
                    "DxResult_embeddings": embedding_bank[DxItem][case_label],
                }
            },
        )
        return case, records, stack
    except Exception:
        stack.close()
        raise


def _vector(value: torch.Tensor) -> torch.Tensor:
    result = value.detach().to(device="cpu", dtype=torch.float32)
    if result.ndim == 1:
        return result
    if result.ndim == 2:
        return result.mean(dim=0)
    if result.ndim > 2:
        return result.reshape(-1)
    raise ValueError(f"Unsupported embedding shape: {tuple(result.shape)}")


def _cosine_scores(
    predicted: torch.Tensor,
    embeddings: Mapping[str, torch.Tensor],
    labels: Sequence[str],
) -> dict[str, float]:
    predicted_vector = _vector(predicted)
    scores: dict[str, float] = {}
    for label in labels:
        if label not in embeddings:
            raise ValueError(f"Embedding bank is missing {label!r}")
        anchor = _vector(embeddings[label])
        if anchor.shape != predicted_vector.shape:
            raise ValueError(
                f"Embedding shape mismatch for {label}: "
                f"{tuple(predicted_vector.shape)} != {tuple(anchor.shape)}"
            )
        scores[label] = float(F.cosine_similarity(predicted_vector, anchor, dim=0).item())
    return scores


def _pseudo_prediction(
    *,
    scores: Mapping[str, float],
    labels: Sequence[str],
    thresholds: Mapping[str, float | None],
    case_label: str,
    case_threshold: float,
    chunk_idx: int,
) -> dict[str, Any]:
    candidates = [
        (label, float(scores[label]))
        for label in labels
        if thresholds[label] is not None and scores[label] >= float(thresholds[label])
    ]
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    top_three = candidates[:3]
    case_rank = labels.index(case_label)
    assigned = next(
        (label for label, _ in top_three if labels.index(label) <= case_rank),
        UNABLE_TO_DETERMINE,
    )
    importance = float(scores[case_label])
    return {
        "chunk_idx": chunk_idx,
        "assigned": assigned,
        # Keep the report-conditioned score at full precision because the
        # zero-candidate policy ranks terminal-layer ROIs by this value.
        "importance": importance,
        "assigned_as_ref": importance >= case_threshold,
        "candidate@top3AUC": {
            label: round(score, 6) for label, score in top_three
        },
        "candidate@ALL": {
            label: round(float(scores[label]), 6) for label in labels
        },
    }


def _chunk_case(source: Case, records: Sequence[_InputRecord]) -> Case:
    result = copy(source)
    result.rois = [record.roi for record in records]
    return result


def run_native_inference(
    *,
    input_payload: dict[str, Any],
    DxItem: str,
    case_label: str,
    support: Any,
    config: Mapping[str, Any],
) -> NativeInferenceResult:
    """Load the real checkpoint and return layered pseudo labels by stable ROI ID."""

    artifacts = resolve_native_artifacts(config)
    options = _native_options(config)
    checkpoint_config = _json_object(artifacts.config_path)
    threshold_bundle = _json_object(artifacts.threshold_bundle_path)
    weight_path = _path(options.get("weight_path"), "backend.model.weight_path")
    embedding_path = _embedding_path(checkpoint_config, options, weight_path)
    _validate_checkpoint_metadata(
        checkpoint_config, threshold_bundle, embedding_path
    )
    cfg = _build_model_config(checkpoint_config, options, weight_path, embedding_path)

    if DxItem not in support.effective_DxItem_list:
        raise ValueError(f"Native CLEE received unsupported DxItem: {DxItem}")
    labels = list(support.active_DxResult_dict[DxItem])
    if DxItem == "Histologic_Type":
        if set(labels) != set(HISTOLOGIC_TYPE_SEVERITY):
            raise ValueError(
                "CLEE Histologic_Type label space does not match the supported "
                "seven-class severity policy"
            )
        labels = list(HISTOLOGIC_TYPE_SEVERITY)
    if case_label not in labels:
        raise ValueError(f"Native CLEE received unsupported {DxItem} result: {case_label}")
    embeddings = load_catalog_embedding_bank(
        embedding_path, cfg.embedding_key, cfg.model_name
    )
    missing_labels = set(labels) - set(embeddings.get(DxItem, {}))
    if missing_labels:
        raise ValueError(f"CLEE embeddings are missing labels: {sorted(missing_labels)}")

    image_root_raw = config.get("image_root")
    image_root = resolve_run_path(image_root_raw) if image_root_raw else None
    source_case, records, resources = _case_records(
        input_payload, DxItem, case_label, cfg, embeddings, image_root
    )
    try:
        from components.person_e.model.modeling_CLEE import CaseLevelEvidenceEvaluator

        model = CaseLevelEvidenceEvaluator.from_config(cfg)
        model.load_shared_prefix(
            str(artifacts.model_checkpoint_path), strict=True, map_location=cfg.device
        )
        model.eval()

        maximum = int(config["max_roi_dxitem_inputs_per_forward"])
        if maximum <= 0:
            raise ValueError("max_roi_dxitem_inputs_per_forward must be positive")
        traces: dict[str, dict[str, Any]] = {record.roi_id: {} for record in records}
        pending = list(records)
        layer_idx = 0
        device_type = torch.device(cfg.device).type
        use_autocast = bool(getattr(cfg, "amp", True)) and device_type == "cuda"
        autocast = (
            torch.autocast(device_type="cuda", dtype=_dtype(options.get("dtype", "bfloat16")))
            if use_autocast
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            while pending:
                chunks = [
                    pending[start : start + maximum]
                    for start in range(0, len(pending), maximum)
                ]
                selected: list[_InputRecord] = []
                for chunk_idx, chunk in enumerate(chunks):
                    outputs = model(case=_chunk_case(source_case, chunk))
                    positions = outputs["prefix_positions"]
                    if len(positions) != len(chunk):
                        raise RuntimeError("CLEE prefix/ROI count mismatch")
                    for record, position in zip(chunk, positions):
                        predicted = outputs["last_hidden"][
                            0, position : position + int(model.prefix_len), :
                        ]
                        scores = _cosine_scores(
                            predicted, embeddings[DxItem], labels
                        )
                        prediction = _pseudo_prediction(
                            scores=scores,
                            labels=labels,
                            thresholds=support.roi_classification_thresholds[DxItem],
                            case_label=case_label,
                            case_threshold=support.threshold_for(
                                DxItem, case_label
                            ),
                            chunk_idx=chunk_idx,
                        )
                        traces[record.roi_id][str(layer_idx)] = {
                            DxItem: prediction
                        }
                        traces[record.roi_id][FINAL_RESULT_KEY] = {
                            DxItem: deepcopy(prediction)
                        }
                        if prediction["assigned_as_ref"]:
                            selected.append(record)
                    del outputs
                if len(chunks) == 1 or not selected or len(selected) == len(pending):
                    break
                pending = selected
                layer_idx += 1
        from components.person_e.adapter import apply_effective_selection

        selection_by_roi_id, selection_summary = apply_effective_selection(
            traces=traces,
            global_idx_by_roi_id={
                record.roi_id: record.roi.global_idx for record in records
            },
            DxItem=DxItem,
            fallback_top_k=support.fallback_top_k,
        )
        return NativeInferenceResult(
            predictions_by_roi_id=traces,
            selection_by_roi_id=selection_by_roi_id,
            selection_summary=selection_summary,
        )
    finally:
        resources.close()
