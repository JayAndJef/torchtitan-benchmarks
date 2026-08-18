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
        from benchmarks.kernel.engine.run import RunOptions, run_kernel_scenario
        from benchmarks.kernel.registry import kernel_scenario_by_name
        from benchmarks.kernel.schema import resolve_shape_and_workload

        scenario = kernel_scenario_by_name(name)
        shape, workload = resolve_shape_and_workload(batch=1)
        result = run_kernel_scenario(
            scenario,
            shape,
            workload,
            # burst_k > 1 deliberately. The backward arms retain their graph
            # and re-run torch.autograd.backward, so a burst re-enters the
            # same compiled backward graph k times in a row. That is the one
            # property of burst timing no CPU test can reach, and this is
            # where it is exercised.
            RunOptions(
                replicates=2,
                samples_per_replicate=3,
                burst_k=2,
                warmup_calls=1,
                memory_iters=1,
            ),
            "smoke",
        )
        failures = [row for row in result.correctness if row.passed is False]
        self.assertEqual(failures, [], f"{name}: correctness gates failed")
        self.assertTrue(result.all_correctness_passed)
        for arm in scenario.arms:
            measured = result.arms[arm.name].modes
            self.assertEqual(set(measured), set(arm.modes), arm.name)
            for mode in arm.modes:
                # Replicate boundaries survive out of the engine, and the
                # pooled view is their concatenation.
                self.assertEqual(len(measured[mode].replicates_us), 2)
                for replicate in measured[mode].replicates_us:
                    self.assertEqual(len(replicate), 3)
                self.assertEqual(len(measured[mode].samples_us), 6)

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
