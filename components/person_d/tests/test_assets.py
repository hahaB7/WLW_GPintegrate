from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from components.person_d.assets import check_digest, required_digest, tree_sha256


class AssetDigestTests(unittest.TestCase):
    def test_tree_hash_follows_the_documented_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b").mkdir()
            (root / "b" / "x.jpg").write_bytes(b"x")
            (root / "B.jpg").write_bytes(b"upper")
            (root / "a.jpg").write_bytes(b"a")
            lines = [
                f"{hashlib.sha256(data).hexdigest()}  {name}"
                for name, data in sorted([("B.jpg", b"upper"), ("a.jpg", b"a"), ("b/x.jpg", b"x")])
            ]
            expected = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
            self.assertEqual((3, expected), tree_sha256(root))

    def test_digests_are_required_and_compared(self) -> None:
        digest = "a" * 64
        self.assertEqual(digest, required_digest({"key": digest}, "key", "matching"))
        for broken in ({}, {"key": None}, {"key": "A" * 64}, {"key": "a" * 63}):
            with self.assertRaisesRegex(ValueError, "requires key"):
                required_digest(broken, "key", "matching")
        check_digest(digest, digest, "asset")
        with self.assertRaisesRegex(ValueError, "asset SHA-256 mismatch"):
            check_digest("b" * 64, digest, "asset")


if __name__ == "__main__":
    unittest.main()
