"""Prepare the shared A.Literature artifact for the WLW pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts.paths import REFERENCE_ROOT, RUN_ROOT, resolve_run_path
from contracts.runtime import validate_artifact


DEFAULT_OUTPUT = RUN_ROOT / "input/pipeline/A_literature.json"
PROFILES = {
    "production": {
        "config": ROOT / "components/person_b/configs/prepare_literature.rag_database.json",
        "source": REFERENCE_ROOT / "person_b/rag_database",
        "image": "wlw/person-b:final-main-42ce738",
    },
    "example": {
        "config": ROOT / "components/person_b/configs/example.json",
        "source": None,
        "image": None,
    },
}


def _container_run_path(path: Path) -> str:
    resolved = resolve_run_path(path)
    return "/run/" + resolved.relative_to(RUN_ROOT.resolve()).as_posix()


def _production_command(output: Path, config: Path, source: Path, image: str) -> list[str]:
    source = source.resolve()
    try:
        source_relative = source.relative_to(REFERENCE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Literature source must stay below {REFERENCE_ROOT.resolve()}; got {source}"
        ) from exc
    config_relative = config.resolve().relative_to(ROOT.resolve())
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-v",
        f"{REFERENCE_ROOT.resolve()}:/reference:ro",
        "-v",
        f"{RUN_ROOT.resolve()}:/run",
        "--entrypoint",
        "python",
        image,
        "-m",
        "components.person_b.prepare_literature",
        "--source",
        "/reference/" + source_relative.as_posix(),
        "--output",
        _container_run_path(output),
        "--config",
        "/app/" + config_relative.as_posix(),
    ]


def _example_command(output: Path, config: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "components.person_b.prepare_literature",
        "--output",
        str(output),
        "--config",
        str(config),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare A.Literature for the WLW pipeline")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="production",
        help="production wraps the real RAG database; example emits synthetic test data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="A.Literature output below sibling run/",
    )
    parser.add_argument("--config", type=Path, help="Override the profile config")
    parser.add_argument("--source", type=Path, help="Override the production RAG database")
    args = parser.parse_args()

    output = resolve_run_path(args.output)
    profile = PROFILES[args.profile]
    config = (args.config or profile["config"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.profile == "production":
        source = (args.source or profile["source"]).resolve()
        image = os.environ.get("WLW_IMAGE_PERSON_B", profile["image"])
        command = _production_command(output, config, source, image)
    else:
        if args.source is not None:
            raise ValueError("--source is not valid with --profile example")
        command = _example_command(output, config)

    print("=== components.person_b.prepare_literature ===", flush=True)
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)

    artifact = json.loads(output.read_text(encoding="utf-8"))
    validate_artifact(artifact, "A.Literature")
    corpus = artifact["payload"]["corpus"]
    section_count = sum(
        len(item["sections"]) for item in artifact["payload"]["literature_list"]
    )
    print(
        f"A.Literature complete: {corpus['literature_count']} literature nodes, "
        f"{section_count} sections, sha256={corpus['sha256']} -> {output}"
    )


if __name__ == "__main__":
    main()
