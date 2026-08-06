"""Regression tests for run completion and profiler-structure validation."""

import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.runner import RunRequest, execute_run
from benchmarks.runtime import CpuPinning


class RunCompletionTests(unittest.TestCase):
    def test_structurally_invalid_traces_leave_run_failed(self) -> None:
        metadata = {
            "requested_gpu": "0",
            "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
            "torch_version": "test",
            "torchtitan_git_rev": "titan-rev",
            "benchmarks_git_rev": "bench-rev",
        }

        def malformed_process(command, **kwargs):
            kwargs["stdout"].write("Training completed\n")
            arm_dir = Path(command[-1])
            for iteration in (20, 40):
                trace = (
                    arm_dir
                    / f"profiling/traces/iteration_{iteration}/rank0_trace.json.gz"
                )
                trace.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(trace, "wt") as trace_file:
                    json.dump({"traceEvents": []}, trace_file)
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.runner.hardware_metadata",
            return_value=("test-gpu", metadata),
        ), mock.patch(
            "benchmarks.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            out_dir = Path(temporary) / "run"
            request = RunRequest(
                gpu="0",
                scenario_name="piper1b_rope",
                arm_name="baseline",
                out_dir=out_dir,
            )
            with self.assertRaisesRegex(RuntimeError, "structural validation"):
                execute_run(
                    request,
                    process_runner=malformed_process,
                    environment={"PATH": os.environ["PATH"]},
                )
            state = json.loads((out_dir / "run_state.json").read_text())

        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["arms"]["baseline"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
