"""Multi-case runner for the contract-first WLW DAG.

The default production profile executes every component in its own pinned
Docker image and mounts only the sibling reference/ and run/ trees.
The example profile remains available for the repository's dependency-free
contract test and runs with the current Python interpreter.
"""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts.paths import (
    REFERENCE_ROOT,
    RUN_ROOT,
    resolve_run_path,
    sibling_logical_path,
)
from contracts.runtime import (
    load_case_list_input,
    validate_artifact,
    validate_case_list_input,
    write_artifact,
    write_case_list_input,
)


INPUTS = RUN_ROOT / "input" / "pipeline"
LOGS = RUN_ROOT / "logs"

PROFILES = {
    "production": {
        "report_decompose": ROOT / "components/person_a/configs/report_decompose.default.json",
        "knowledge_retrieval": ROOT / "components/person_b/configs/dense.default.json",
        "query_generation": ROOT / "components/person_a/configs/query_generation.production.json",
        "interest_pattern": ROOT / "components/person_c/configs/region_proposal.json",
        "visual_filter": ROOT / "components/person_d/configs/native.json",
        "clee": ROOT / "components/person_e/configs/native.json",
    },
    "example": {
        "report_decompose": ROOT / "components/person_a/configs/example.json",
        "knowledge_retrieval": ROOT / "components/person_b/configs/example.json",
        "query_generation": ROOT / "components/person_a/configs/example.json",
        "interest_pattern": ROOT / "components/person_c/configs/example.json",
        "visual_filter": ROOT / "components/person_d/configs/example.json",
        "clee": ROOT / "components/person_e/configs/default.json",
    },
}

COMPONENTS = {
    "report_decompose": {
        "module": "components.person_a.report_decompose",
        "image_env": "WLW_IMAGE_PERSON_A",
        "image": "wlw/person-a:0.10.0-lab23-production",
        "gpu": False,
    },
    "interest_pattern": {
        "module": "components.person_c.interest_pattern",
        "image_env": "WLW_IMAGE_PERSON_C",
        "image": "wlw/person-c:0.1.0-lab23-production",
        "gpu": True,
    },
    "knowledge_retrieval": {
        "module": "components.person_b.knowledge_retrieval",
        "image_env": "WLW_IMAGE_PERSON_B",
        "image": "wlw/person-b:0.5.0-lab23-production",
        "gpu": True,
    },
    "query_generation": {
        "module": "components.person_a.query_generation",
        "image_env": "WLW_IMAGE_PERSON_A",
        "image": "wlw/person-a:0.10.0-lab23-production",
        "gpu": True,
    },
    "visual_filter": {
        "module": "components.person_d.visual_filter",
        "image_env": "WLW_IMAGE_PERSON_D",
        "image": "wlw/person-d:0.2.0-lab23-production",
        "gpu": True,
    },
    "clee": {
        "module": "components.person_e.clee",
        "image_env": "WLW_IMAGE_PERSON_E",
        "image": "wlw/person-e:1.1.0-lab23-production",
        "gpu": True,
    },
}

STRUCTURED_REPORT_PRODUCER = "pipeline/structured-report-adapter:1.0.0"
STRUCTURED_REPORT_RAW_TEXT_PLACEHOLDER = (
    "Structured report supplied; raw report unavailable."
)
PARENTHETICAL_ABBREVIATION = re.compile(r"\(([A-Za-z][A-Za-z0-9_-]*)\)\s*$")
SKIPPABLE_CASE_ERROR_MARKERS = {
    "interest_pattern": (
        ("torch.OutOfMemoryError:", "torch.OutOfMemoryError"),
        ("torch.cuda.OutOfMemoryError:", "torch.cuda.OutOfMemoryError"),
    ),
    "query_generation": (
        ("LearnableQueryGenerationError:", "LearnableQueryGenerationError"),
    ),
}


def _case_directory_name(case_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("._")
    if not safe:
        raise ValueError(f"case_id cannot be represented as a directory: {case_id!r}")
    return safe


def _abbreviate_dx_result(value: object, *, context: str) -> str:
    """Return a trailing parenthetical abbreviation, or an existing class."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    result = value.strip()
    match = PARENTHETICAL_ABBREVIATION.search(result)
    return match.group(1) if match else result


def _load_structured_case_list_input(path: Path) -> dict:
    """Load a CaseList whose cases additionally carry structured reports."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Structured CaseList input must be a JSON object")
    cases = document.get("case_list")
    if not isinstance(cases, list):
        raise ValueError("Structured CaseList input requires a case_list array")

    requested_items = document.get("DxItem_list")
    if not isinstance(requested_items, list):
        raise ValueError("Structured CaseList input requires a DxItem_list array")
    requested_set = set(requested_items)

    canonical_source = deepcopy(document)
    for index, (case, canonical_case) in enumerate(
        zip(cases, canonical_source["case_list"], strict=True)
    ):
        if not isinstance(case, dict):
            raise ValueError(f"case_list[{index}] must be an object")
        case_id = case.get("case_id", f"index {index}")
        structured_report = case.get("structured_report")
        if not isinstance(structured_report, dict):
            raise ValueError(
                f"Case {case_id!r} requires an object structured_report when "
                "--has-structured-report is set"
            )
        dx_items = structured_report.get("DxItems")
        if not isinstance(dx_items, dict):
            raise ValueError(
                f"Case {case_id!r} structured_report requires an object DxItems"
            )
        actual_set = set(dx_items)
        if actual_set != requested_set:
            raise ValueError(
                f"Case {case_id!r} structured_report DxItems do not match "
                f"DxItem_list; missing={sorted(requested_set - actual_set)}, "
                f"unexpected={sorted(actual_set - requested_set)}"
            )
        for dx_item, record in dx_items.items():
            if not isinstance(record, dict):
                raise ValueError(
                    f"Case {case_id!r} DxItem {dx_item!r} must be an object"
                )
            _abbreviate_dx_result(
                record.get("DxResultCls"),
                context=f"Case {case_id!r} DxItem {dx_item!r} DxResultCls",
            )
        if canonical_case.get("report_raw_content") == "":
            canonical_case["report_raw_content"] = (
                STRUCTURED_REPORT_RAW_TEXT_PLACEHOLDER
            )
        canonical_case.pop("structured_report", None)

    validate_case_list_input(canonical_source)
    return document


def _container_run_path(path: Path) -> str:
    resolved = resolve_run_path(path)
    return "/run/" + resolved.relative_to(RUN_ROOT.resolve()).as_posix()


def _container_config_path(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Component config does not exist: {resolved}")
    relative = resolved.relative_to(ROOT.resolve())
    return "/app/" + relative.as_posix()


def _load_profile_configs(profile: str) -> dict[str, Path]:
    """Validate host-side configs before starting any component container."""

    configs = PROFILES[profile]
    documents: dict[str, dict] = {}
    for stage, path in configs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{profile} config for {stage} is missing: {path}")
        documents[stage] = json.loads(path.read_text(encoding="utf-8"))

    if profile == "production":
        checks = {
            "knowledge_retrieval": documents["knowledge_retrieval"].get("mode") == "dense",
            "query_generation": (
                documents["query_generation"].get("backend", {}).get("mode")
                == "learnable_soft_prompt"
                and documents["query_generation"].get("backend", {}).get(
                    "device_map_strategy"
                )
                == "single_device"
                and documents["query_generation"].get("backend", {}).get(
                    "retry_with_canonical_dx"
                )
                is True
            ),
            "interest_pattern": documents["interest_pattern"].get("mode") == "region_proposal",
            "visual_filter": (
                documents["visual_filter"].get("mode") == "native"
                and documents["visual_filter"].get("matching", {}).get("enable") is True
            ),
            "clee": documents["clee"].get("backend", {}).get("mode") == "native",
        }
        invalid = sorted(stage for stage, valid in checks.items() if not valid)
        if invalid:
            raise ValueError(
                "Production profile contains a non-production or bypass config for: "
                + ", ".join(invalid)
            )
    return configs


def _resolve_source_wsi(raw_path: str, case_list_path: Path) -> Path:
    """Resolve a CaseList WSI path without depending on the caller's cwd."""

    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        source_relative = (case_list_path.parent / candidate).resolve()
        project_relative = (ROOT / candidate).resolve()
        if source_relative.is_file():
            resolved = source_relative
        elif project_relative.is_file():
            resolved = project_relative
        else:
            raise FileNotFoundError(
                f"WSI path {raw_path!r} was not found relative to "
                f"{case_list_path.parent} or {ROOT}"
            )
    if not resolved.is_file():
        raise FileNotFoundError(f"WSI does not exist: {resolved}")
    return resolved


def _docker_wsi_location(resolved: Path) -> tuple[str, tuple[Path, str] | None]:
    """Return the container path and optional read-only external directory mount."""

    try:
        relative = resolved.relative_to(RUN_ROOT.resolve())
    except ValueError:
        source_directory = resolved.parent
        directory_key = sha256(
            source_directory.as_posix().encode("utf-8")
        ).hexdigest()[:16]
        target_directory = f"/external_wsi/{directory_key}"
        return (
            f"{target_directory}/{resolved.name}",
            (source_directory, target_directory),
        )
    return "/run/" + relative.as_posix(), None


def _external_wsi_mounts(
    raw_case: dict,
    case_list_path: Path,
) -> list[tuple[Path, str]]:
    mounts: dict[str, Path] = {}
    for block in raw_case["tissue_blocks"]:
        for stain in block["stains"]:
            resolved = _resolve_source_wsi(stain["filepath"], case_list_path)
            _, mount = _docker_wsi_location(resolved)
            if mount is None:
                continue
            source, target = mount
            previous = mounts.setdefault(target, source)
            if previous != source:
                raise ValueError(
                    f"External WSI mount collision at {target}: "
                    f"{previous} and {source}"
                )
    return [(source, target) for target, source in sorted(mounts.items())]


def _materialize_case(
    source: dict,
    raw_case: dict,
    case_list_path: Path,
    *,
    executor: str,
) -> dict:
    """Return a single-case input with WSI paths valid in the chosen runtime."""

    case = deepcopy(raw_case)
    if (
        "structured_report" in case
        and case.get("report_raw_content") == ""
    ):
        case["report_raw_content"] = STRUCTURED_REPORT_RAW_TEXT_PLACEHOLDER
    # Component CaseList inputs intentionally exclude downstream metadata.
    case.pop("structured_report", None)
    if executor != "docker":
        return {"DxItem_list": source["DxItem_list"], "case_list": [case]}
    for block in case["tissue_blocks"]:
        for stain in block["stains"]:
            resolved = _resolve_source_wsi(stain["filepath"], case_list_path)
            stain["filepath"], _ = _docker_wsi_location(resolved)
    return {"DxItem_list": source["DxItem_list"], "case_list": [case]}


def _reference_wsi_ids(record: dict, case: dict) -> list[str]:
    explicit = record.get("referenceWSI")
    if explicit is not None:
        if not isinstance(explicit, list):
            raise ValueError("structured_report referenceWSI must be an array")
        return list(explicit)

    reference_types = record.get("referenceType", [])
    if not isinstance(reference_types, list):
        raise ValueError("structured_report referenceType must be an array")
    allowed = {str(item).upper() for item in reference_types}
    return [
        stain["stain_id"]
        for block in case["tissue_blocks"]
        for stain in block["stains"]
        if not allowed or str(stain["stain_type"]).upper() in allowed
    ]


def _reference_blocks(reference_wsi: list[str], case: dict) -> list[str] | None:
    selected = set(reference_wsi)
    blocks = list(
        dict.fromkeys(
            str(block["block_id"])
            for block in case["tissue_blocks"]
            if str(block["block_id"])
            and any(stain["stain_id"] in selected for stain in block["stains"])
        )
    )
    return blocks or None


def _build_structured_report_artifact(
    materialized_source: dict,
    source_report: dict,
    case_list_path: Path,
) -> dict:
    """Build D.DxPairs without invoking report_decompose."""

    case = deepcopy(materialized_source["case_list"][0])
    case_id = case["case_id"]
    records: dict[str, dict] = {}
    source_records = source_report["DxItems"]
    for index, dx_item in enumerate(materialized_source["DxItem_list"], start=1):
        source_record = source_records[dx_item]
        raw_result = source_record["DxResultCls"].strip()
        result = _abbreviate_dx_result(
            raw_result,
            context=f"Case {case_id!r} DxItem {dx_item!r} DxResultCls",
        )
        reference_wsi = _reference_wsi_ids(source_record, case)
        reference_block = source_record.get("referenceBlock")
        if "referenceBlock" not in source_record:
            reference_block = _reference_blocks(reference_wsi, case)

        record = {
            "dx_pair_id": source_record.get(
                "dx_pair_id", f"{case_id}-structured-dx-{index:03d}"
            ),
            "source_report_id": source_record.get(
                "source_report_id", f"{case_id}-structured-report"
            ),
            "DxResultCls": result,
            "DxResultTxt": source_record.get("DxResultTxt", raw_result),
            "DxResultRawTxt": source_record.get("DxResultRawTxt"),
            "referenceBlock": reference_block,
            "referenceType": deepcopy(source_record.get("referenceType", [])),
            "referenceWSI": reference_wsi,
            "appear_reportTypes": deepcopy(
                source_record.get("appear_reportTypes", [])
            ),
            "optionTypes": source_record.get("optionTypes", "provided"),
            "has_numericalData": source_record.get("has_numericalData", False),
        }
        if result != raw_result:
            record["raw_DxResultCls"] = raw_result
        records[dx_item] = record

    case["structured_report"] = {
        "reportType": source_report.get("reportType"),
        "reportSubTypes": deepcopy(source_report.get("reportSubTypes", [])),
        "DxItems": records,
    }
    for block in case["tissue_blocks"]:
        for stain in block["stains"]:
            stain["roi_num"] = 0
            stain["roi_list"] = []

    return {
        "contract": "D.DxPairs",
        "schema_version": "2.0",
        "artifact_id": f"D-structured-{case_id}",
        "case_id": case_id,
        "producer": STRUCTURED_REPORT_PRODUCER,
        "payload": {
            "data_mode": "inference",
            "DxItem_list": list(materialized_source["DxItem_list"]),
            "reference_versions": {
                "structured_report_input": {
                    "source_path": sibling_logical_path(case_list_path),
                    "result_class_normalization": (
                        "trailing_parenthetical_abbreviation"
                    ),
                }
            },
            "case_list": [case],
        },
    }


def _docker_command(
    stage: str,
    inputs: list[Path],
    output: Path,
    config: Path,
    external_wsi_mounts: list[tuple[Path, str]],
) -> list[str]:
    component = COMPONENTS[stage]
    image = os.environ.get(component["image_env"], component["image"])
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--shm-size",
        "8g",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-v",
        f"{REFERENCE_ROOT.resolve()}:/reference:ro",
        "-v",
        f"{RUN_ROOT.resolve()}:/run",
    ]
    for source, target in external_wsi_mounts:
        command.extend(
            [
                "--mount",
                f"type=bind,source={source},target={target},readonly",
            ]
        )
    if component["gpu"]:
        command.extend(
            [
                "--gpus",
                "all",
                "-e",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            ]
        )
    command.extend(
        [
            "--entrypoint",
            "python",
            image,
            "-m",
            component["module"],
        ]
    )
    for input_path in inputs:
        command.extend(["--input", _container_run_path(input_path)])
    command.extend(
        [
            "--output",
            _container_run_path(output),
            "--config",
            _container_config_path(config),
        ]
    )
    return command


def _local_command(
    stage: str,
    inputs: list[Path],
    output: Path,
    config: Path,
) -> list[str]:
    command = [sys.executable, "-m", COMPONENTS[stage]["module"]]
    for input_path in inputs:
        command.extend(["--input", str(input_path)])
    command.extend(["--output", str(output), "--config", str(config)])
    return command


def _run_logged(command: list[str], log: TextIO) -> None:
    rendered = " ".join(command)
    print(f"$ {rendered}", flush=True)
    log.write(f"$ {rendered}\n")
    log.flush()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    captured_output: deque[str] = deque(maxlen=200)
    for line in process.stdout:
        captured_output.append(line)
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(
            return_code,
            command,
            output="".join(captured_output),
        )


def _skippable_case_error(
    stage: str,
    error: subprocess.CalledProcessError,
) -> tuple[str, str] | None:
    """Return ``(error_type, reason)`` for explicitly skippable case errors."""

    markers = SKIPPABLE_CASE_ERROR_MARKERS.get(stage, ())
    if not markers or not isinstance(error.output, str):
        return None
    for line in reversed(error.output.splitlines()):
        for marker, error_type in markers:
            if marker in line:
                return error_type, line.split(marker, 1)[1].strip()
    return None


def run_component(
    stage: str,
    inputs: list[Path],
    output: Path,
    config: Path,
    *,
    executor: str,
    log: TextIO,
    external_wsi_mounts: list[tuple[Path, str]] | None = None,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = (
        _docker_command(
            stage,
            inputs,
            output,
            config,
            external_wsi_mounts or [],
        )
        if executor == "docker"
        else _local_command(stage, inputs, output, config)
    )
    print(f"\n=== {COMPONENTS[stage]['module']} ===", flush=True)
    started = time.monotonic()
    _run_logged(command, log)
    elapsed = round(time.monotonic() - started, 3)
    return {
        "stage": stage,
        "module": COMPONENTS[stage]["module"],
        "config": config.relative_to(ROOT).as_posix(),
        "image": (
            os.environ.get(COMPONENTS[stage]["image_env"], COMPONENTS[stage]["image"])
            if executor == "docker"
            else None
        ),
        "elapsed_seconds": elapsed,
    }


def _assert_production_literature(path: Path) -> None:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    validate_artifact(artifact, "A.Literature")
    corpus = artifact["payload"]["corpus"]
    if artifact["producer"].endswith("example-stub:0.1.0") or corpus["corpus_id"] == "example-corpus":
        raise ValueError(
            "Production pipeline refuses the synthetic A.Literature artifact. "
            "Run pipeline/run_prepare_literature.py first."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the WLW pipeline once per CaseList case")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="production",
        help="production uses native configs; example is contract-test only",
    )
    parser.add_argument(
        "--executor",
        choices=("docker", "local"),
        help="default: docker for production, local for example",
    )
    parser.add_argument(
        "--case-list",
        type=Path,
        default=INPUTS / "cases.json",
        help="CaseList input below the sibling run directory",
    )
    parser.add_argument(
        "--literature",
        type=Path,
        default=INPUTS / "A_literature.json",
        help="Shared A.Literature artifact used for every case",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=RUN_ROOT / "output" / "pipeline",
        help="Output directory below sibling run/",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Run only this case_id; repeat to select multiple cases",
    )
    parser.add_argument(
        "--has-structured-report",
        action="store_true",
        help=(
            "skip report_decompose and build D.DxPairs from each case's "
            "structured_report.DxItems"
        ),
    )
    args = parser.parse_args()

    executor = args.executor or ("docker" if args.profile == "production" else "local")
    if args.profile == "production" and executor != "docker":
        raise ValueError("The production profile requires --executor docker")

    configs = _load_profile_configs(args.profile)
    case_list_path = resolve_run_path(args.case_list)
    literature = resolve_run_path(args.literature)
    artifacts = resolve_run_path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    if args.profile == "production":
        _assert_production_literature(literature)
    source = (
        _load_structured_case_list_input(case_list_path)
        if args.has_structured_report
        else load_case_list_input(case_list_path)
    )

    requested = set(args.case_ids or [])
    known = {case["case_id"] for case in source["case_list"]}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown --case-id values: {sorted(unknown)}")

    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOGS / f"pipeline-{args.profile}-{stamp}.log"
    completed_cases = []
    skipped_cases = []
    execution_records = []
    used_directories: set[str] = set()

    with log_path.open("w", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "profile": args.profile,
                    "executor": executor,
                    "case_list": sibling_logical_path(case_list_path),
                    "literature": sibling_logical_path(literature),
                    "has_structured_report": args.has_structured_report,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for raw_case in source["case_list"]:
            case_id = raw_case["case_id"]
            if requested and case_id not in requested:
                continue
            directory_name = _case_directory_name(case_id)
            if directory_name in used_directories:
                raise ValueError(f"Case directory collision after normalization: {case_id!r}")
            used_directories.add(directory_name)

            case_artifacts = artifacts / "cases" / directory_name
            case_input = case_artifacts / "case_input.json"
            d = case_artifacts / "D_dx_pairs.json"
            e = case_artifacts / "E_rois.json"
            f = case_artifacts / "F_chunks.json"
            g = case_artifacts / "G_queries.json"
            h = case_artifacts / "H_matches.json"
            selected_rois = case_artifacts / "I_selected_rois.json"
            external_wsi_mounts = (
                _external_wsi_mounts(raw_case, case_list_path)
                if executor == "docker"
                else []
            )
            materialized_source = _materialize_case(
                source,
                raw_case,
                case_list_path,
                executor=executor,
            )
            write_case_list_input(
                materialized_source,
                case_input,
            )

            print(f"\n##### case {case_id} #####", flush=True)
            if args.has_structured_report:
                print(
                    "\n=== structured report -> D.DxPairs "
                    "(report_decompose skipped) ===",
                    flush=True,
                )
                write_artifact(
                    _build_structured_report_artifact(
                        materialized_source,
                        raw_case["structured_report"],
                        case_list_path,
                    ),
                    d,
                    "D.DxPairs",
                )
                log.write(
                    f"case {case_id}: built D.DxPairs from structured_report; "
                    "report_decompose skipped\n"
                )
                log.flush()

            stages = [
                ("interest_pattern", [case_input], e),
                ("knowledge_retrieval", [literature, d], f),
                ("query_generation", [d, f], g),
                ("visual_filter", [e, g], h),
                ("clee", [d, h], selected_rois),
            ]
            if not args.has_structured_report:
                stages.insert(0, ("report_decompose", [case_input], d))
            case_execution = []
            case_skipped = False
            for stage, inputs, output in stages:
                try:
                    record = run_component(
                        stage,
                        inputs,
                        output,
                        configs[stage],
                        executor=executor,
                        log=log,
                        external_wsi_mounts=external_wsi_mounts,
                    )
                except subprocess.CalledProcessError as exc:
                    skip_error = _skippable_case_error(stage, exc)
                    if skip_error is None:
                        raise
                    error_type, reason = skip_error
                    skipped_record = {
                        "case_id": case_id,
                        "failed_stage": stage,
                        "error_type": error_type,
                        "reason": reason,
                        "case_input_path": case_input.relative_to(
                            artifacts
                        ).as_posix(),
                        "elapsed_seconds_before_skip": round(
                            sum(
                                item["elapsed_seconds"]
                                for item in case_execution
                            ),
                            3,
                        ),
                    }
                    skipped_cases.append(skipped_record)
                    skip_message = (
                        f"Skipping case {case_id}: {stage} raised "
                        f"{error_type}: {reason}"
                    )
                    print(f"\n{skip_message}", flush=True)
                    log.write(skip_message + "\n")
                    log.flush()
                    case_skipped = True
                    break
                record["case_id"] = case_id
                case_execution.append(record)
                execution_records.append(record)

            if case_skipped:
                continue

            completed_cases.append(
                {
                    "case_id": case_id,
                    "case_input_path": case_input.relative_to(artifacts).as_posix(),
                    "selected_rois_path": selected_rois.relative_to(artifacts).as_posix(),
                    "elapsed_seconds": round(
                        sum(item["elapsed_seconds"] for item in case_execution), 3
                    ),
                }
            )

    run_manifest = {
        "profile": args.profile,
        "executor": executor,
        "has_structured_report": args.has_structured_report,
        "case_list": sibling_logical_path(case_list_path),
        "literature_artifact": sibling_logical_path(literature),
        "log_path": sibling_logical_path(log_path),
        "completed_cases": completed_cases,
        "skipped_cases": skipped_cases,
        "executions": execution_records,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    run_manifest_path = artifacts / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\nPipeline complete: {len(completed_cases)} completed, "
        f"{len(skipped_cases)} skipped. "
        f"Run manifest: {run_manifest_path}\nLog: {log_path}"
    )


if __name__ == "__main__":
    main()
