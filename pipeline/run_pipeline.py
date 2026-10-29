"""Multi-case runner for the contract-first WLW DAG.

The default production profile executes every component in its own pinned
Docker image and mounts only the sibling reference/ and run/ trees.
The example profile remains available for the repository's dependency-free
contract test and runs with the current Python interpreter.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
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
    write_case_list_input,
)


INPUTS = RUN_ROOT / "input" / "pipeline"
LOGS = RUN_ROOT / "logs"

PROFILES = {
    "production": {
        "report_decompose": ROOT / "components/person_a/configs/report_decompose.default.json",
        "knowledge_retrieval": ROOT / "components/person_b/configs/dense.default.json",
        "query_generation": ROOT / "components/person_a/configs/query_generation.default.json",
        "interest_pattern": ROOT / "components/person_c/configs/default.json",
        "visual_filter": ROOT / "components/person_d/configs/native.example.json",
        "clee": ROOT / "components/person_e/configs/native.json",
    },
    "example": {
        "report_decompose": ROOT / "components/person_a/configs/example.json",
        "knowledge_retrieval": ROOT / "components/person_b/configs/example.json",
        "query_generation": ROOT / "components/person_a/configs/example.json",
        "interest_pattern": ROOT / "components/person_c/configs/demo.json",
        "visual_filter": ROOT / "components/person_d/configs/example.json",
        "clee": ROOT / "components/person_e/configs/default.json",
    },
}

COMPONENTS = {
    "report_decompose": {
        "module": "components.person_a.report_decompose",
        "image_env": "WLW_IMAGE_PERSON_A",
        "image": "wlw/person-a:lab22-integrated",
        "gpu": False,
    },
    "interest_pattern": {
        "module": "components.person_c.interest_pattern",
        "image_env": "WLW_IMAGE_PERSON_C",
        "image": "wlw/person-c:0.4.0-final-20261027",
        "gpu": True,
    },
    "knowledge_retrieval": {
        "module": "components.person_b.knowledge_retrieval",
        "image_env": "WLW_IMAGE_PERSON_B",
        "image": "wlw/person-b:final-main-42ce738",
        "gpu": True,
    },
    "query_generation": {
        "module": "components.person_a.query_generation",
        "image_env": "WLW_IMAGE_PERSON_A",
        "image": "wlw/person-a:lab22-integrated",
        "gpu": True,
    },
    "visual_filter": {
        "module": "components.person_d.visual_filter",
        "image_env": "WLW_IMAGE_PERSON_D",
        "image": "wlw/person-d:lab22-integrated",
        "gpu": True,
    },
    "clee": {
        "module": "components.person_e.clee",
        "image_env": "WLW_IMAGE_PERSON_E",
        "image": "wlw/person-e:final-main-42ce738",
        "gpu": True,
    },
}


def _case_directory_name(case_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id).strip("._")
    if not safe:
        raise ValueError(f"case_id cannot be represented as a directory: {case_id!r}")
    return safe


def _container_run_path(path: Path) -> str:
    resolved = resolve_run_path(path)
    return "/run/" + resolved.relative_to(RUN_ROOT.resolve()).as_posix()


def _container_config_path(path: Path) -> str:
    resolved = path.resolve()
    relative = resolved.relative_to(ROOT.resolve())
    return "/app/" + relative.as_posix()


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


def _materialize_case(
    source: dict,
    raw_case: dict,
    case_list_path: Path,
    *,
    executor: str,
) -> dict:
    """Return a single-case input with WSI paths valid in the chosen runtime."""

    case = deepcopy(raw_case)
    if executor != "docker":
        return {"DxItem_list": source["DxItem_list"], "case_list": [case]}
    for block in case["tissue_blocks"]:
        for stain in block["stains"]:
            resolved = _resolve_source_wsi(stain["filepath"], case_list_path)
            try:
                relative = resolved.relative_to(RUN_ROOT.resolve())
            except ValueError as exc:
                raise ValueError(
                    "Docker production execution requires every WSI below "
                    f"{RUN_ROOT.resolve()}; got {resolved}"
                ) from exc
            stain["filepath"] = "/run/" + relative.as_posix()
    return {"DxItem_list": source["DxItem_list"], "case_list": [case]}


def _docker_command(
    stage: str,
    inputs: list[Path],
    output: Path,
    config: Path,
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
    if component["gpu"]:
        command.extend(["--gpus", "all"])
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
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_component(
    stage: str,
    inputs: list[Path],
    output: Path,
    config: Path,
    *,
    executor: str,
    log: TextIO,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = (
        _docker_command(stage, inputs, output, config)
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
    args = parser.parse_args()

    executor = args.executor or ("docker" if args.profile == "production" else "local")
    if args.profile == "production" and executor != "docker":
        raise ValueError("The production profile requires --executor docker")

    configs = PROFILES[args.profile]
    case_list_path = resolve_run_path(args.case_list)
    literature = resolve_run_path(args.literature)
    artifacts = resolve_run_path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    if args.profile == "production":
        _assert_production_literature(literature)
    source = load_case_list_input(case_list_path)

    requested = set(args.case_ids or [])
    known = {case["case_id"] for case in source["case_list"]}
    unknown = requested - known
    if unknown:
        raise ValueError(f"Unknown --case-id values: {sorted(unknown)}")

    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOGS / f"pipeline-{args.profile}-{stamp}.log"
    completed_cases = []
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
            write_case_list_input(
                _materialize_case(
                    source,
                    raw_case,
                    case_list_path,
                    executor=executor,
                ),
                case_input,
            )

            print(f"\n##### case {case_id} #####", flush=True)
            stages = [
                ("report_decompose", [case_input], d),
                ("interest_pattern", [case_input], e),
                ("knowledge_retrieval", [literature, d], f),
                ("query_generation", [d, f], g),
                ("visual_filter", [e, g], h),
                ("clee", [d, h], selected_rois),
            ]
            case_execution = []
            for stage, inputs, output in stages:
                record = run_component(
                    stage,
                    inputs,
                    output,
                    configs[stage],
                    executor=executor,
                    log=log,
                )
                record["case_id"] = case_id
                case_execution.append(record)
                execution_records.append(record)

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
        "case_list": sibling_logical_path(case_list_path),
        "literature_artifact": sibling_logical_path(literature),
        "log_path": sibling_logical_path(log_path),
        "completed_cases": completed_cases,
        "executions": execution_records,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    run_manifest_path = artifacts / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\nPipeline complete: {len(completed_cases)} cases. "
        f"Run manifest: {run_manifest_path}\nLog: {log_path}"
    )


if __name__ == "__main__":
    main()
