"""Read-only loader for CLEE per-DxResult embedding catalogs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch


EMBEDDING_CATALOG_SCHEMA_VERSION = "clee_dxitem_embedding_catalog_v1"
EMBEDDING_MANIFEST_FILENAME = "embedding_manifest.json"
EMBEDDING_TYPES = ("pretrain", "inference")


def _filename_component(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Embedding filename components must be non-empty")
    return quote(value, safe=" ()_-.+")


def embedding_filename(embedding_type: str, DxItem: str, DxResult: str) -> str:
    if embedding_type not in EMBEDDING_TYPES:
        raise ValueError(f"Unsupported embedding type: {embedding_type!r}")
    return (
        f"{embedding_type}_"
        f"[{_filename_component(DxItem)}]"
        f"[{_filename_component(DxResult)}]_embeddings.pt"
    )


def normalize_embedding_key(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) and value:
        return (value,)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(isinstance(item, str) and item for item in value)
    ):
        return tuple(value)
    raise ValueError(f"Invalid CLEE embedding_key: {value!r}")


def load_embedding_manifest(directory: Path) -> dict[str, Any]:
    if not directory.is_dir():
        raise NotADirectoryError(
            f"CLEE embedding catalog must be a directory: {directory}"
        )
    manifest_path = directory / EMBEDDING_MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Embedding manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Embedding manifest must be a JSON object")
    if manifest.get("schema_version") != EMBEDDING_CATALOG_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported embedding manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )

    item_order = manifest.get("dx_item_order")
    result_order = manifest.get("dx_result_order")
    entries = manifest.get("entries")
    if not isinstance(item_order, list) or not item_order or not all(
        isinstance(item, str) and item for item in item_order
    ):
        raise ValueError("Embedding manifest dx_item_order is invalid")
    if len(set(item_order)) != len(item_order):
        raise ValueError("Embedding manifest dx_item_order contains duplicates")
    if not isinstance(result_order, dict) or set(result_order) != set(item_order):
        raise ValueError(
            "Embedding manifest dx_result_order must match dx_item_order"
        )

    ordered_pairs: list[tuple[str, str]] = []
    for DxItem in item_order:
        labels = result_order[DxItem]
        if not isinstance(labels, list) or not labels or not all(
            isinstance(label, str) and label for label in labels
        ):
            raise ValueError(
                f"Embedding manifest labels are invalid for {DxItem!r}"
            )
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"Embedding manifest labels contain duplicates for {DxItem!r}"
            )
        ordered_pairs.extend((DxItem, label) for label in labels)

    if not isinstance(entries, list):
        raise ValueError("Embedding manifest entries must be an array")
    by_pair: dict[tuple[str, str], dict[str, str]] = {}
    seen_filenames: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Embedding manifest entries must be objects")
        embedding_type = entry.get("type")
        DxItem = entry.get("dx_item")
        DxResult = entry.get("dx_result")
        filename = entry.get("filename")
        if not all(
            isinstance(value, str) and value
            for value in (embedding_type, DxItem, DxResult, filename)
        ):
            raise ValueError("Embedding manifest entry fields are invalid")
        expected = embedding_filename(embedding_type, DxItem, DxResult)
        if filename != expected:
            raise ValueError(
                f"Embedding filename mismatch: {filename!r} != {expected!r}"
            )
        pair = (DxItem, DxResult)
        if pair in by_pair or filename in seen_filenames:
            raise ValueError(f"Duplicate embedding manifest entry: {pair!r}")
        by_pair[pair] = {
            "type": embedding_type,
            "dx_item": DxItem,
            "dx_result": DxResult,
            "filename": filename,
        }
        seen_filenames.add(filename)
    if set(ordered_pairs) != set(by_pair) or len(ordered_pairs) != len(by_pair):
        raise ValueError(
            "Embedding manifest ordering and entries describe different pairs"
        )
    manifest["entries"] = [by_pair[pair] for pair in ordered_pairs]
    return manifest


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Embedding payload must be an object: {path}")
    return payload


def _resolve_embedding(
    content: Mapping[str, Any],
    selector: tuple[str, ...],
    model_name: str | None,
) -> Any:
    if len(selector) == 1:
        return content[selector[0]]
    if not model_name:
        raise ValueError("model_name is required for nested embedding keys")
    prompt_type, embedding_type = selector
    return content[prompt_type]["embedding"][model_name][embedding_type]


def load_embedding_bank(
    directory: Path,
    embedding_key: Any,
    model_name: str | None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Load a complete catalog without mutating the reference tree."""

    selector = normalize_embedding_key(embedding_key)
    manifest = load_embedding_manifest(directory)
    bank: dict[str, dict[str, torch.Tensor]] = {}
    for entry in manifest["entries"]:
        path = directory / entry["filename"]
        if not path.is_file():
            raise FileNotFoundError(f"Embedding file is missing: {path}")
        payload = _load_mapping(path)
        if payload.get("type") != entry["type"]:
            raise ValueError(f"Embedding type mismatch: {path}")
        content = payload.get("content")
        if not isinstance(content, Mapping) or not content:
            raise ValueError(f"Embedding content is invalid: {path}")
        try:
            raw_embedding = _resolve_embedding(content, selector, model_name)
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Embedding selector {selector!r} is unavailable in {path}"
            ) from error
        tensor = (
            raw_embedding
            if torch.is_tensor(raw_embedding)
            else torch.as_tensor(raw_embedding)
        )
        bank.setdefault(entry["dx_item"], {})[entry["dx_result"]] = (
            tensor.detach().cpu()
        )
    if not bank:
        raise ValueError(f"CLEE embedding catalog is empty: {directory}")
    return bank
