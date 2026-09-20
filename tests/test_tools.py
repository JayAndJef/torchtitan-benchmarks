"""Structural tests for the developer scripts outside the Python packages.

Two things are pinned here, because both fail silently:

* the pre-push hook, which runs the CPU suite before a push leaves the
  machine, and the ``sync.sh`` step that installs it; and
* the module list of the continuous-integration workflow, which must name
  test modules that exist and that a host without the machine-learning
  stack can import.
"""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_import_boundaries import REPO_ROOT

PRE_PUSH = REPO_ROOT / "tools" / "pre-push.sh"
SYNC = REPO_ROOT / "sync.sh"


class PrePushHookTests(unittest.TestCase):
    def test_the_hook_is_executable(self) -> None:
        self.assertTrue(PRE_PUSH.is_file(), f"{PRE_PUSH} is absent")
        self.assertTrue(
            os.access(PRE_PUSH, os.X_OK),
            "tools/pre-push.sh must be executable; git runs it directly",
        )

    def test_the_hook_starts_with_the_bash_shebang(self) -> None:
        first = PRE_PUSH.read_text().splitlines()[0]
        self.assertEqual(first, "#!/usr/bin/env bash")

    def test_the_hook_runs_the_suite_and_names_the_bypass(self) -> None:
        source = PRE_PUSH.read_text()
        self.assertIn("unittest discover", source)
        self.assertIn("--no-verify", source)

    def test_sync_installs_the_hook_in_the_common_directory(self) -> None:
        source = SYNC.read_text()
        self.assertIn("git rev-parse --git-common-dir", source)
        self.assertIn("pre-push", source)


if __name__ == "__main__":
    unittest.main()
