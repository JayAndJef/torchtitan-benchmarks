"""GPU smoke test: every kernel scenario builds, runs, and passes its gates.

Skips itself without CUDA; the rope scenario additionally needs g++ >= 13 for
the TransformerEngine extension build.
"""

import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch

    CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - torch import failure means no GPU tests
    CUDA = False


def _has_gcc_13() -> bool:
    try:
        version = subprocess.check_output(
            ["g++", "-dumpfullversion"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    return int(version.split(".")[0]) >= 13


@unittest.skipUnless(CUDA, "kernel benchmarks require CUDA")
class KernelScenarioSmokeTests(unittest.TestCase):
    def _run(self, name: str) -> None:
        from benchmarks.kernel_bench import RunOptions, run_kernel_scenario
        from benchmarks.kernels import Piper1BSpec, kernel_scenario_by_name

        scenario = kernel_scenario_by_name(name)
        result = run_kernel_scenario(
            scenario,
            Piper1BSpec(batch=1),
            RunOptions(n=3, warmup=1, memory_iters=1),
            "smoke",
        )
        failures = [row for row in result.correctness if row.passed is False]
        self.assertEqual(failures, [], f"{name}: correctness gates failed")
        self.assertTrue(result.all_correctness_passed)
        for arm in scenario.arms:
            measured = result.arms[arm.name].modes
            self.assertEqual(set(measured), set(arm.modes), arm.name)
            for mode in arm.modes:
                self.assertEqual(len(measured[mode].samples_us), 3)

    def test_swiglu(self) -> None:
        self._run("swiglu")

    def test_qkv(self) -> None:
        self._run("qkv")

    def test_lm_head(self) -> None:
        self._run("lm_head")

    @unittest.skipUnless(_has_gcc_13(), "TE extension needs g++ >= 13")
    def test_rope(self) -> None:
        self._run("rope")


if __name__ == "__main__":
    unittest.main()
