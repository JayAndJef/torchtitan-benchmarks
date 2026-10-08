"""Tests for the cuDNN identity every manifest records, and the refusal of a mixed cuDNN load.

TransformerEngine links the cuDNN parts by soname and opens ``libcudnn.so``
by name, so the loader can bind a system cuDNN for TE beside the wheel's
cuDNN that torch loads. ``torch.backends.cudnn.version()`` then raises, and
an arm that imports TE and calls ``varlen_attn`` dies. ``cudnn_env.sh``,
which ``run_bench.sh`` sources, puts the wheel's cuDNN first; the harness
refuses a run whose TE process would still map a second cuDNN. The
cuBLASLt version gets the provenance tests too, because a reader must see
which cuBLASLt a run bound.
"""

import ctypes.util
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.execution import provenance

_UNAVAILABLE = re.compile(r"^unavailable: ")

REPO_ROOT = Path(__file__).resolve().parent.parent

CUDNN_ENV = REPO_ROOT / "cudnn_env.sh"
"""The script that ``run_bench.sh`` sources to put torch's cuDNN first."""

VENV_CUDNN = "/venv/site-packages/nvidia/cudnn/lib"
"""The cuDNN directory of torch in the fake probe outputs."""


def probe_output(loaded: list[str], runtime: str = "92400") -> str:
    """One fake ``CUDNN_LOADER_PROBE`` output, after a warning that ``run_text`` merged in."""
    probe = {
        "bundled": [VENV_CUDNN],
        "loaded": loaded,
        "build": "92400",
        "runtime": runtime,
    }
    return "UserWarning: CUDA initialization\n" + json.dumps(probe)


class CudnnProvenanceTests(unittest.TestCase):
    def test_the_manifest_records_both_cudnn_identities(self) -> None:
        """Both fields reach ``hardware_metadata``, under stable names."""
        with (
            mock.patch.object(provenance, "run_text", return_value="x"),
            mock.patch.object(
                provenance, "_cudnn_loader_resolves", return_value=VENV_CUDNN
            ),
            mock.patch.object(provenance, "_megatron_git_rev", return_value="r"),
            mock.patch.object(provenance, "_te_version", return_value="v"),
        ):
            _, metadata = provenance.hardware_metadata(
                mock.Mock(), "0", "some-hardware"
            )
        self.assertIn("cudnn_torch_build", metadata)
        self.assertIn("cudnn_loader_resolves", metadata)

    def test_the_two_cudnn_fields_are_collected_separately(self) -> None:
        """Neither field is derived from the other.

        A single field could not express the split this exists to record.
        """
        with (
            mock.patch.object(
                provenance, "_cudnn_torch_build", return_value="9.24.0"
            ) as build,
            mock.patch.object(
                provenance, "_cudnn_loader_resolves", return_value=VENV_CUDNN
            ) as loader,
            mock.patch.object(provenance, "run_text", return_value="x"),
            mock.patch.object(provenance, "_megatron_git_rev", return_value="r"),
            mock.patch.object(provenance, "_te_version", return_value="v"),
        ):
            _, metadata = provenance.hardware_metadata(mock.Mock(), "0", "label")
        build.assert_called_once_with()
        loader.assert_called_once_with()
        self.assertEqual(metadata["cudnn_torch_build"], "9.24.0")
        self.assertEqual(metadata["cudnn_loader_resolves"], VENV_CUDNN)

    def test_collecting_the_torch_build_never_raises(self) -> None:
        """A broken probe reports itself instead of failing the run."""
        with mock.patch.object(
            provenance, "run_text", return_value="unavailable: boom"
        ):
            self.assertRegex(provenance._cudnn_torch_build(), _UNAVAILABLE)

    def test_an_unavailable_loader_probe_does_not_raise(self) -> None:
        """A probe that fails, such as on a host without TransformerEngine, does not stop the run."""
        with mock.patch.object(
            provenance, "run_text", return_value="unavailable: boom"
        ):
            self.assertRegex(provenance._cudnn_loader_resolves(), _UNAVAILABLE)

    def test_run_text_turns_a_missing_command_into_a_diagnostic(self) -> None:
        """The property both probes rely on, pinned once."""
        self.assertRegex(
            provenance.run_text(["definitely-not-a-command-here"]), _UNAVAILABLE
        )


class CudnnRefusalTests(unittest.TestCase):
    """The refusal of a cuDNN that TransformerEngine maps beside torch's own."""

    def test_torch_cudnn_alone_records_its_directory(self) -> None:
        self.assertEqual(
            provenance.cudnn_loader_resolves(probe_output([VENV_CUDNN])),
            VENV_CUDNN,
        )

    def test_a_second_cudnn_directory_raises(self) -> None:
        output = probe_output([VENV_CUDNN, "/usr/lib64"])
        with self.assertRaisesRegex(ValueError, "cuDNN in /usr/lib64 beside"):
            provenance.cudnn_loader_resolves(output)

    def test_a_version_query_that_raises_raises(self) -> None:
        output = probe_output([VENV_CUDNN], runtime="raises: incompatible")
        with self.assertRaisesRegex(ValueError, "raises: incompatible"):
            provenance.cudnn_loader_resolves(output)

    def test_a_runtime_newer_than_the_build_raises(self) -> None:
        """torch accepts a newer minor runtime, but the run would not measure torch's cuDNN."""
        output = probe_output([VENV_CUDNN], runtime="92500")
        with self.assertRaisesRegex(ValueError, "'92500' against the build 92400"):
            provenance.cudnn_loader_resolves(output)

    def test_an_output_without_a_json_line_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no JSON line"):
            provenance.cudnn_loader_resolves("Segmentation fault")

    def test_the_metadata_refuses_a_second_cudnn(self) -> None:
        def fake_run_text(command, **kwargs):
            if command[-1] == provenance.CUDNN_LOADER_PROBE:
                return probe_output([VENV_CUDNN, "/usr/lib64"])
            return "0, NVIDIA H200, GPU-a, 570.211.01"

        with mock.patch.object(provenance, "run_text", fake_run_text):
            with self.assertRaisesRegex(ValueError, "beside torch's cuDNN"):
                provenance.hardware_metadata(mock.Mock(), "0", "auto")


def unfixed_environment() -> dict[str, str]:
    """This process's environment without ``CUDNN_HOME`` and without a wheel cuDNN directory on ``LD_LIBRARY_PATH``."""
    environment = {
        key: value for key, value in os.environ.items() if key != "CUDNN_HOME"
    }
    entries = environment.get("LD_LIBRARY_PATH", "").split(":")
    environment["LD_LIBRARY_PATH"] = ":".join(
        entry for entry in entries if entry and "nvidia/cudnn" not in entry
    )
    return environment


class CudnnLoadTests(unittest.TestCase):
    """The loader probe in a real TransformerEngine process, without and with ``cudnn_env.sh``."""

    def _probe(self, command: list[str]) -> dict:
        result = subprocess.run(
            command,
            env=unfixed_environment(),
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout.splitlines()[-1])

    def test_the_unfixed_load_maps_a_second_cudnn(self) -> None:
        if ctypes.util.find_library("cudnn") is None:
            self.skipTest("the loader knows no system cuDNN, so the load cannot mix")
        probe = self._probe([sys.executable, "-c", provenance.CUDNN_LOADER_PROBE])
        foreign = [path for path in probe["loaded"] if path not in probe["bundled"]]
        self.assertTrue(
            foreign or probe["runtime"] != probe["build"],
            f"no mixed load without cudnn_env.sh: {probe}",
        )
        with self.assertRaisesRegex(ValueError, "beside torch's cuDNN"):
            provenance.cudnn_loader_resolves(json.dumps(probe))

    def test_cudnn_env_maps_torch_cudnn_alone(self) -> None:
        probe = self._probe(
            [
                "bash",
                "-c",
                'set -e; source "$1"; shift; exec "$@"',
                "bash",
                str(CUDNN_ENV),
                sys.executable,
                "-c",
                provenance.CUDNN_LOADER_PROBE,
            ]
        )
        self.assertEqual(probe["runtime"], probe["build"])
        self.assertEqual(probe["loaded"], probe["bundled"])
        self.assertEqual(len(probe["bundled"]), 1)
        self.assertTrue(
            Path(probe["bundled"][0]).is_relative_to(Path(sys.prefix).resolve()),
            probe["bundled"],
        )
        self.assertEqual(
            provenance.cudnn_loader_resolves(json.dumps(probe)), probe["bundled"][0]
        )


class CublasltProvenanceTests(unittest.TestCase):
    def test_the_manifest_records_the_cublaslt_version(self) -> None:
        """The probe result reaches ``hardware_metadata`` under a stable name."""
        with (
            mock.patch.object(
                provenance,
                "_cublaslt_version",
                return_value="130101 /venv/libcublasLt.so.13",
            ) as probe,
            mock.patch.object(provenance, "run_text", return_value="x"),
            mock.patch.object(
                provenance, "_cudnn_loader_resolves", return_value=VENV_CUDNN
            ),
            mock.patch.object(provenance, "_megatron_git_rev", return_value="r"),
            mock.patch.object(provenance, "_te_version", return_value="v"),
        ):
            _, metadata = provenance.hardware_metadata(mock.Mock(), "0", "label")
        probe.assert_called_once_with()
        self.assertEqual(
            metadata["cublaslt_version"], "130101 /venv/libcublasLt.so.13"
        )

    def test_collecting_the_cublaslt_version_never_raises(self) -> None:
        with mock.patch.object(
            provenance, "run_text", return_value="unavailable: boom"
        ):
            self.assertRegex(provenance._cublaslt_version(), _UNAVAILABLE)

    def test_the_cublaslt_probe_reports_a_version_and_a_path_on_this_host(
        self,
    ) -> None:
        """The probe gives the integer version and a real library, or says why it could not."""
        resolved = provenance._cublaslt_version()
        if _UNAVAILABLE.match(resolved):
            self.skipTest(f"no cuBLASLt resolvable here: {resolved}")
        version, path = resolved.split(" ", 1)
        self.assertRegex(version, r"^[0-9]{6}$")
        self.assertTrue(Path(path).exists(), path)
        self.assertIn("libcublasLt", Path(path).name)


if __name__ == "__main__":
    unittest.main()
