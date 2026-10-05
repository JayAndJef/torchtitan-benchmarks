"""Tests for the cuDNN identity every manifest now records.

TransformerEngine's ``DT_NEEDED`` entries carry no ``RUNPATH`` and torch
loads its own cuDNN lazily, so the loader can bind a system cuDNN for TE
while torch expects the wheel's. Which cuDNN a megatron arm runs is then a
property of the host. These tests pin that the two versions reach the
manifest as separate fields, and that collecting them can never fail a run.
The cuBLASLt version gets the same tests, because TE's grouped GEMM needs 13.4+.
"""

import re
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.execution import provenance

_UNAVAILABLE = re.compile(r"^unavailable: ")


class CudnnProvenanceTests(unittest.TestCase):
    def test_the_manifest_records_both_cudnn_identities(self) -> None:
        """Both fields reach ``hardware_metadata``, under stable names."""
        with (
            mock.patch.object(provenance, "run_text", return_value="x"),
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
                provenance,
                "_cudnn_loader_resolves",
                return_value="/usr/lib64/libcudnn.so.9.23.2",
            ) as loader,
            mock.patch.object(provenance, "run_text", return_value="x"),
            mock.patch.object(provenance, "_megatron_git_rev", return_value="r"),
            mock.patch.object(provenance, "_te_version", return_value="v"),
        ):
            _, metadata = provenance.hardware_metadata(mock.Mock(), "0", "label")
        build.assert_called_once_with()
        loader.assert_called_once_with()
        self.assertEqual(metadata["cudnn_torch_build"], "9.24.0")
        self.assertEqual(
            metadata["cudnn_loader_resolves"], "/usr/lib64/libcudnn.so.9.23.2"
        )

    def test_collecting_the_torch_build_never_raises(self) -> None:
        """A broken probe reports itself instead of failing the run."""
        with mock.patch.object(
            provenance, "run_text", return_value="unavailable: boom"
        ):
            self.assertRegex(provenance._cudnn_torch_build(), _UNAVAILABLE)

    def test_collecting_the_loader_resolution_never_raises(self) -> None:
        with mock.patch.object(
            provenance, "run_text", return_value="unavailable: boom"
        ):
            self.assertRegex(provenance._cudnn_loader_resolves(), _UNAVAILABLE)

    def test_run_text_turns_a_missing_command_into_a_diagnostic(self) -> None:
        """The property both probes rely on, pinned once."""
        self.assertRegex(
            provenance.run_text(["definitely-not-a-command-here"]), _UNAVAILABLE
        )

    def test_the_loader_probe_reports_a_real_path_on_this_host(self) -> None:
        """The probe resolves a library, or says why it could not."""
        resolved = provenance._cudnn_loader_resolves()
        if _UNAVAILABLE.match(resolved):
            self.skipTest(f"no cuDNN resolvable here: {resolved}")
        self.assertTrue(Path(resolved).exists(), resolved)
        self.assertIn("libcudnn", Path(resolved).name)


class CublasltProvenanceTests(unittest.TestCase):
    def test_the_manifest_records_the_cublaslt_version(self) -> None:
        """The probe result reaches ``hardware_metadata`` under a stable name."""
        with (
            mock.patch.object(
                provenance,
                "_cublaslt_version",
                return_value="130401 /venv/libcublasLt.so.13",
            ) as probe,
            mock.patch.object(provenance, "run_text", return_value="x"),
            mock.patch.object(provenance, "_megatron_git_rev", return_value="r"),
            mock.patch.object(provenance, "_te_version", return_value="v"),
        ):
            _, metadata = provenance.hardware_metadata(mock.Mock(), "0", "label")
        probe.assert_called_once_with()
        self.assertEqual(
            metadata["cublaslt_version"], "130401 /venv/libcublasLt.so.13"
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
