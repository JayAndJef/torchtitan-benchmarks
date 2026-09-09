"""The engine knowledge base under ``database/`` checks and builds.

``database/`` is gitignored, so both tests skip on a checkout that does not
hold it. The skip is per method, not per module: the test census loads this
module with ``loadTestsFromName`` and counts its tests, and a module-level
skip would turn that count into a lie.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATABASE = REPO_ROOT / "database"
TOOLS = REPO_ROOT / "tools"
ABSENT = "database/ is gitignored and absent on this checkout"


def run_tool(name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOLS / name), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


class DatabaseTests(unittest.TestCase):
    @unittest.skipUnless(DATABASE.is_dir(), ABSENT)
    def test_check_passes(self) -> None:
        done = run_tool("db_check.py")
        self.assertEqual(done.returncode, 0, f"db_check.py failed:\n{done.stdout}{done.stderr}")

    @unittest.skipUnless(DATABASE.is_dir(), ABSENT)
    def test_doc_builds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "kb.html"
            done = run_tool("db_build_doc.py", "--out", str(out))
            self.assertEqual(done.returncode, 0, f"db_build_doc.py failed:\n{done.stdout}{done.stderr}")
            self.assertTrue(out.is_file())
            self.assertIn("<h1", out.read_text())


if __name__ == "__main__":
    unittest.main()
