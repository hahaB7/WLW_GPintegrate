"""Integrity checks for the external assets of Person C.

The expected digests are pinned here (the same values are listed in ``external-assets.yaml``).
``verify_assets`` runs before any model is loaded; a mismatch raises ``AssetError`` and the CLI
exits non-zero without writing an E artifact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

CONCH_REPO = "https://github.com/mahmoodlab/CONCH"
CONCH_REVISION = "141cc09c7d4ff33d8eda562bd75169b457f71a62"
# sha256 over the sorted lines "<file-sha256>  <relative path>" of third_party/CONCH, the unmodified
# conch/open_clip_custom/ and LICENSE of CONCH_REVISION (__pycache__ is ignored).
CONCH_LIBRARY_TREE_SHA256 = "9e6419e1c78f79406acf635529bfde1c4029f15acb97221d11e76345d187febd"
CONCH_CHECKPOINT_SHA256 = "40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c"
BCSS_VOCAB_SHA256 = "47ea2b920bb73dfd43e9f3b9892fab20b80e6812902a9b914b6986f13166aa62"


class AssetError(RuntimeError):
    """An external asset is missing or does not match its pinned digest."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    lines = sorted(
        f"{file_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _check(label: str, path: Path, expected: str, actual: str) -> None:
    if actual != expected:
        raise AssetError(
            f"{label} digest mismatch for {path}: expected sha256 {expected}, got {actual}"
        )


def verify_assets(region: Mapping[str, str]) -> None:
    """Check the resolved ``conch_lib_path`` (shipped in the component), ``conch_checkpoint`` and ``vocab_json`` (external)."""

    lib, checkpoint, vocab = (Path(region[k]) for k in
                              ("conch_lib_path", "conch_checkpoint", "vocab_json"))
    if not lib.is_dir():
        raise AssetError(f"CONCH library is not a directory: {lib}")
    for label, path in (("CONCH checkpoint", checkpoint), ("BCSS vocabulary", vocab)):
        if not path.is_file():
            raise AssetError(f"{label} is missing: {path}")
    _check("CONCH library", lib, CONCH_LIBRARY_TREE_SHA256, tree_sha256(lib))
    _check("CONCH checkpoint", checkpoint, CONCH_CHECKPOINT_SHA256, file_sha256(checkpoint))
    _check("BCSS vocabulary", vocab, BCSS_VOCAB_SHA256, file_sha256(vocab))
