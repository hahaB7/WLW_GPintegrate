"""Gemma 3 condition soft-prompt inference for Person A query generation.

This module is the production inference boundary distilled from the Nano5
training/evaluation programs.  It deliberately does not contain the training
loop, patient reports, the base model, or the learned checkpoint.  Those large
or controlled assets stay outside the repository and are mounted at runtime.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from typing import Any

from components.person_a.asset_integrity import verify_external_asset
from components.person_a.local_model_integrity import verify_local_model
from contracts.paths import resolve_person_reference_path

SOFT_PROMPT_MARKER = "<SOFT_PROMPT_INSERT>"
TARGET_FORMAT = "visual_attribute_option_conditions_v3"
CONDITION_LABELS = {
    "Must_True",
    "Must_False",
    "High_Possibly_True",
    "Low_Possibly_True",
    "Negligible",
    "Not_Mentioned",
}


class LearnableQueryGenerationError(ValueError):
    """Raised when model inputs or generated attributes are not usable."""


def _flatten(document: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in document.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten(value, path))
        else:
            flattened[path] = value
    return flattened


def _node_types(document: dict[str, Any], prefix: str = "") -> dict[str, str]:
    types: dict[str, str] = {}
    for key, value in document.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        types[path] = "object" if isinstance(value, dict) else "leaf"
        if isinstance(value, dict):
            types.update(_node_types(value, path))
    return types


def parse_strict_json_object(text: str) -> dict[str, Any]:
    """Require the complete generation to be exactly one JSON object."""

    stripped = str(text or "").strip()
    if not stripped:
        raise LearnableQueryGenerationError("Model generated an empty response")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        context_start = max(0, exc.pos - 80)
        context_end = min(len(stripped), exc.pos + 80)
        context = stripped[context_start:context_end]
        raise LearnableQueryGenerationError(
            f"Model response is not strict JSON: {exc.msg} "
            f"at line {exc.lineno} column {exc.colno}; "
            f"response context={context!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise LearnableQueryGenerationError(
            f"Model response root must be an object, not {type(parsed).__name__}"
        )
    return parsed


def normalize_generated_attributes(
    generated: dict[str, Any],
    template: dict[str, Any],
    allowed_values: dict[str, set[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project model output onto the canonical condition schema.

    Unexpected model nodes are discarded.  A missing canonical condition, a
    node with the wrong structural type, or an illegal condition value is
    replaced with ``Not_Mentioned``.  The returned diagnostics preserve every
    repair decision for artifact provenance.
    """

    expected_nodes = _node_types(template)
    actual_nodes = _node_types(generated)
    missing = sorted(set(expected_nodes) - set(actual_nodes))
    unexpected = sorted(set(actual_nodes) - set(expected_nodes))
    wrong_type = sorted(
        path
        for path in set(expected_nodes) & set(actual_nodes)
        if expected_nodes[path] != actual_nodes[path]
    )
    missing_value = object()
    filled_condition_paths: list[str] = []
    invalid_value_paths: list[str] = []
    preserved_condition_count = 0

    def normalize_node(expected: Any, actual: Any, prefix: str) -> Any:
        nonlocal preserved_condition_count
        if isinstance(expected, dict):
            actual_mapping = actual if isinstance(actual, dict) else {}
            return {
                key: normalize_node(
                    expected_value,
                    actual_mapping.get(key, missing_value),
                    f"{prefix}.{key}" if prefix else str(key),
                )
                for key, expected_value in expected.items()
            }

        allowed = allowed_values.get(prefix)
        if not allowed or "Not_Mentioned" not in allowed:
            raise ValueError(
                f"Canonical condition schema has no Not_Mentioned fallback at "
                f"{prefix}"
            )
        if (
            actual is not missing_value
            and not isinstance(actual, dict)
            and isinstance(actual, str)
            and actual in allowed
        ):
            preserved_condition_count += 1
            return actual

        filled_condition_paths.append(prefix)
        if actual is not missing_value and not isinstance(actual, dict):
            invalid_value_paths.append(prefix)
        return "Not_Mentioned"

    normalized = normalize_node(template, generated, "")
    diagnostics = {
        "policy": "drop_unexpected_fill_unusable_not_mentioned",
        "missing_nodes": missing,
        "unexpected_nodes": unexpected,
        "wrong_type_nodes": wrong_type,
        "invalid_value_paths": sorted(invalid_value_paths),
        "filled_condition_paths": sorted(filled_condition_paths),
        "missing_node_count": len(missing),
        "dropped_node_count": len(unexpected),
        "wrong_type_node_count": len(wrong_type),
        "invalid_value_count": len(invalid_value_paths),
        "filled_condition_count": len(filled_condition_paths),
        "preserved_condition_count": preserved_condition_count,
    }
    return normalized, diagnostics


def _chunk_text(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        content = str(chunk.get("text", "") or "").strip()
        if "Content:" in content:
            content = content.split("Content:", 1)[1].strip()
        if not content:
            raise LearnableQueryGenerationError(
                f"Retrieved chunk {index} has no text"
            )
        parts.append(f"<chunk{index}>\n{content}\n</chunk{index}>")
    return "\n\n".join(parts)


def _user_text(dx_text: str, chunks: list[dict[str, Any]]) -> str:
    """Keep this prompt byte-for-byte aligned with condition-model training."""

    return (
        f"DxText:\n{dx_text}\n\n"
        f"ChunkText:\n{_chunk_text(chunks)}\n\n"
        f"Task:\n"
        f"Generate the visual attribute condition JSON from DxText and the relevant retrieved chunks.\n"
        f"Some retrieved chunks may be unrelated to DxText. Use relevant chunks and ignore unrelated chunks.\n"
        f"For every canonical attribute, include ALL canonical options, each mapped to exactly one condition string.\n"
        f"Allowed conditions: Must_True, Must_False, High_Possibly_True, Low_Possibly_True, Negligible, Not_Mentioned.\n"
        f"For options supported by the relevant chunks, output the subtype-specific condition learned during training.\n"
        f"Predict qualitative occurrence likelihood categories, not merely which options are mentioned.\n"
        f"For options not supported by the relevant chunks, output Not_Mentioned.\n"
        f"Merge evidence across relevant chunks; unrelated chunks must never supply attribute conditions.\n"
        f"Condition labels describe subtype-level occurrence rules, not observations in this patient's tissue.\n"
        f"Use the canonical nested group/attribute/option schema learned during training.\n"
        f"Return the JSON object only, with no markdown, explanation, metadata, options list, or type fields.\n"
        f"Start directly with {{.\n\n"
        f"Attribute:\n{SOFT_PROMPT_MARKER}"
    )


class LearnableSoftPromptGenerator:
    """Load Gemma and one learned prompt, then generate validated attributes."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        condition_schema: dict[str, Any],
        template_provenance: dict[str, Any],
    ):
        self.config = deepcopy(config)
        self.model_name = str(config["model_name"])
        self.model_revision = config.get("model_revision") or None
        self.model_path = resolve_person_reference_path("person_a", config["model_path"])
        self.checkpoint_path = resolve_person_reference_path(
            "person_a", config["checkpoint_path"]
        )
        if not isinstance(condition_schema, dict) or not condition_schema:
            raise ValueError("condition_schema must be a non-empty object")
        template_sha256 = template_provenance.get("sha256")
        if not isinstance(template_sha256, str) or len(template_sha256) != 64:
            raise ValueError("template_provenance needs a SHA-256")
        self.attribute_template = deepcopy(condition_schema)
        self.allowed_values = {
            leaf_path: set(CONDITION_LABELS)
            for leaf_path in _flatten(self.attribute_template)
        }
        self.template_provenance = deepcopy(template_provenance)
        self.required_chunk_count = int(config.get("required_chunk_count", 3))
        self.asset_integrity = str(config.get("asset_integrity", "enforce"))
        self.max_input_len = int(config.get("max_input_len", 8192))
        self.max_new_tokens = int(config.get("max_new_tokens", 768))
        self.require_cuda = bool(config.get("require_cuda", True))
        self.dtype_name = str(config.get("dtype", "auto"))
        self.device_map_strategy = str(
            config.get("device_map_strategy", "single_device")
        )
        self.gpu_memory_limit = str(
            config.get("gpu_memory_limit", "1200MiB")
        )
        self.cpu_memory_limit = str(
            config.get("cpu_memory_limit", "12GiB")
        )

        if self.required_chunk_count < 1:
            raise ValueError("required_chunk_count must be positive")
        if self.device_map_strategy not in {
            "single_device",
            "auto_cpu_offload",
        }:
            raise ValueError(
                "device_map_strategy must be single_device or auto_cpu_offload"
            )
        checkpoint_integrity = verify_external_asset(
            self.checkpoint_path,
            integrity_mode=self.asset_integrity,
        )
        self.provenance = {
            "backend": "learnable_soft_prompt",
            "model_name": self.model_name,
            "model_id": self.model_name,
            "model_revision": self.model_revision,
            "checkpoint_name": self.checkpoint_path.name,
            "checkpoint_sha256": checkpoint_integrity["sha256"],
            "checkpoint_integrity": checkpoint_integrity,
            "visual_attribute_template": self.template_provenance,
            "required_chunk_count": self.required_chunk_count,
            "target_format": TARGET_FORMAT,
            "device_map_strategy": self.device_map_strategy,
        }
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._soft_prompt: Any = None
        self._soft_prompt_len: int | None = None
        self._device: str | None = None
        self.last_generation_diagnostics: dict[str, Any] = {}

    def _dtype(self, torch: Any, device: str) -> Any:
        if self.dtype_name == "auto":
            if device == "cpu":
                return torch.float32
            return (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        choices = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.dtype_name not in choices:
            raise ValueError("dtype must be auto, float16, bfloat16 or float32")
        return choices[self.dtype_name]

    def _load(self) -> None:
        if self._model is not None:
            return
        integrity = verify_local_model(
            self.model_path, model_id=self.model_name, revision=self.model_revision,
        )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Learnable query generation requires torch and transformers"
            ) from exc

        cuda_available = torch.cuda.is_available()
        if self.require_cuda and not cuda_available:
            raise RuntimeError(
                "Learnable query generation requires CUDA, but no GPU is visible"
            )
        device = "cuda" if cuda_available else "cpu"
        dtype = self._dtype(torch, device)
        common: dict[str, Any] = {"local_files_only": True}
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), **common)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            **common,
        }
        if device == "cuda" and self.device_map_strategy == "auto_cpu_offload":
            # Docker Desktop on a 4 GiB display GPU can expose less allocatable
            # CUDA memory than nvidia-smi reports. Keep the original BF16 model
            # precision, but dispatch only a bounded share of its layers to the
            # GPU and retain the rest in host RAM.
            model_kwargs.update(
                {
                    "device_map": "auto",
                    "max_memory": {
                        0: self.gpu_memory_limit,
                        "cpu": self.cpu_memory_limit,
                    },
                    "low_cpu_mem_usage": True,
                    "offload_buffers": True,
                }
            )
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **model_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                **model_kwargs,
            ).to(device)
        model.config.use_cache = True
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        soft_prompt = checkpoint.get("soft_prompt")
        if soft_prompt is None or soft_prompt.ndim != 2:
            raise LearnableQueryGenerationError(
                "Checkpoint must contain a two-dimensional soft_prompt tensor"
            )
        checkpoint_model = checkpoint.get("model_name")
        if checkpoint_model and checkpoint_model != self.model_name:
            raise LearnableQueryGenerationError(
                f"Checkpoint model {checkpoint_model!r} does not match "
                f"configured model {self.model_name!r}"
            )
        checkpoint_revision = checkpoint.get("model_revision")
        if checkpoint_revision and checkpoint_revision != self.model_revision:
            raise LearnableQueryGenerationError(
                "Checkpoint model revision does not match the verified local model"
            )
        if checkpoint.get("target_format") != TARGET_FORMAT:
            raise LearnableQueryGenerationError(
                "Checkpoint target_format does not match the condition backend"
            )
        if checkpoint.get("template_sha256") != self.template_provenance["sha256"]:
            raise LearnableQueryGenerationError(
                "Checkpoint and visual attribute template do not match"
            )
        checkpoint_conditions = checkpoint.get("condition_labels")
        if checkpoint_conditions is not None and set(checkpoint_conditions) != CONDITION_LABELS:
            raise LearnableQueryGenerationError(
                "Checkpoint condition labels do not match the runtime vocabulary"
            )
        embedding_size = model.get_input_embeddings().embedding_dim
        if soft_prompt.shape[1] != embedding_size:
            raise LearnableQueryGenerationError(
                f"Checkpoint embedding width {soft_prompt.shape[1]} does not "
                f"match model width {embedding_size}"
            )
        soft_prompt_len = int(
            checkpoint.get("soft_prompt_len", soft_prompt.shape[0])
        )
        if soft_prompt_len != soft_prompt.shape[0]:
            raise LearnableQueryGenerationError(
                "Checkpoint soft_prompt_len does not match its tensor"
            )

        self.provenance.update(
            {
                "model_integrity": integrity,
                "checkpoint_run_name": checkpoint.get("run_name"),
                "checkpoint_epoch": checkpoint.get("current_epoch"),
                "soft_prompt_len": soft_prompt_len,
                "gpu_memory_limit": (
                    self.gpu_memory_limit
                    if device == "cuda"
                    and self.device_map_strategy == "auto_cpu_offload"
                    else None
                ),
                "model_device_map": {
                    str(name): str(location)
                    for name, location in getattr(
                        model, "hf_device_map", {}
                    ).items()
                },
            }
        )
        embedding_device = str(model.get_input_embeddings().weight.device)
        if embedding_device == "meta":
            raise RuntimeError("Model input embeddings remained on the meta device")
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._soft_prompt = soft_prompt.to(
            device=embedding_device,
            dtype=model.get_input_embeddings().weight.dtype,
        )
        self._soft_prompt_len = soft_prompt_len
        self._device = embedding_device

    def _split_prompt(self, user_text: str) -> dict[str, Any]:
        torch = self._torch
        tokenizer = self._tokenizer
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            add_generation_prompt=True,
            tokenize=False,
        )
        if SOFT_PROMPT_MARKER not in rendered:
            raise LearnableQueryGenerationError(
                "Tokenizer chat template lost the soft-prompt insertion marker"
            )
        prefix_text, suffix_text = rendered.split(SOFT_PROMPT_MARKER, 1)
        prefix_full = tokenizer(
            prefix_text,
            add_special_tokens=False,
            padding=False,
            return_tensors="pt",
        )
        suffix = tokenizer(
            suffix_text,
            add_special_tokens=False,
            padding=False,
            return_tensors="pt",
        )
        suffix_len = suffix["input_ids"].shape[-1]
        max_prefix_len = max(
            1,
            self.max_input_len - suffix_len - int(self._soft_prompt_len),
        )
        old_side = tokenizer.truncation_side
        tokenizer.truncation_side = "left"
        try:
            prefix = tokenizer(
                prefix_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_prefix_len,
                padding=False,
                return_tensors="pt",
            )
        finally:
            tokenizer.truncation_side = old_side
        if "attention_mask" not in prefix:
            prefix["attention_mask"] = torch.ones_like(prefix["input_ids"])
        if "attention_mask" not in suffix:
            suffix["attention_mask"] = torch.ones_like(suffix["input_ids"])
        return {"prefix": prefix, "suffix": suffix}

    def generate(
        self,
        dx_text: str,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not str(dx_text or "").strip():
            raise LearnableQueryGenerationError("DxText is empty")
        if len(chunks) != self.required_chunk_count:
            raise LearnableQueryGenerationError(
                f"The trained soft prompt requires exactly "
                f"{self.required_chunk_count} chunks, received {len(chunks)}"
            )
        self._load()
        torch = self._torch
        model = self._model
        tokenizer = self._tokenizer
        device = self._device
        encoded = self._split_prompt(_user_text(str(dx_text).strip(), chunks))
        prefix_ids = encoded["prefix"]["input_ids"].to(device)
        prefix_mask = encoded["prefix"]["attention_mask"].to(device)
        suffix_ids = encoded["suffix"]["input_ids"].to(device)
        suffix_mask = encoded["suffix"]["attention_mask"].to(device)
        embedding = model.get_input_embeddings()
        model_dtype = embedding.weight.dtype
        prefix_embeds = embedding(prefix_ids).to(dtype=model_dtype)
        suffix_embeds = embedding(suffix_ids).to(dtype=model_dtype)
        soft_embeds = self._soft_prompt.unsqueeze(0).expand(
            prefix_ids.size(0), -1, -1
        )
        inputs_embeds = torch.cat(
            [prefix_embeds, soft_embeds, suffix_embeds], dim=1
        )
        soft_mask = torch.ones(
            prefix_ids.size(0),
            int(self._soft_prompt_len),
            dtype=prefix_mask.dtype,
            device=device,
        )
        attention_mask = torch.cat(
            [prefix_mask, soft_mask, suffix_mask], dim=1
        )
        with torch.inference_mode():
            generated_ids = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_text = tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        self.last_generation_diagnostics = {
            "response_sha256": hashlib.sha256(
                generated_text.encode("utf-8")
            ).hexdigest(),
            "response_chars": len(generated_text),
            "generated_tokens": int(generated_ids.shape[-1]),
        }
        generated = parse_strict_json_object(generated_text)
        normalized, normalization_diagnostics = normalize_generated_attributes(
            generated,
            self.attribute_template,
            self.allowed_values,
        )
        self.last_generation_diagnostics["schema_normalization"] = (
            normalization_diagnostics
        )
        return normalized
