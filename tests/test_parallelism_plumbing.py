"""The parallelism axis threaded through the harness, without leaving one GPU.

``tests/test_parallelism.py`` covers the axis itself -- the degrees, the
schedules and the fifteen validator rules. This module covers the path the
value takes: the ``<gpu>`` positional read as a device set, the five CLI
options, ``RunRequest``, ``_resolve_run``, the child environment, the
provenance query, the NUMA walk, and manifest schema 10.

**The properties under test are mostly negative.** At the trivial spec every
recorded fact and every environment variable has to be the one this repo has
always produced, character for character, because ``--resume`` compares
several of them in every directory under ``out/`` and roughly one hundred
manifests record the ``<gpu>`` string. The command lines are pinned next to
the other goldens in ``tests/test_migration_contract.py``.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from click.testing import CliRunner

from benchmarks.artifacts.manifests import (
    MANIFEST_SCHEMA_VERSION,
    _resume_mismatches,
    manifest_data,
)
from benchmarks.cli.e2e import _execution_options, run_command
from benchmarks.cli.main import cli
from benchmarks.e2e.parallelism import (
    ParallelismSpec,
    TRIVIAL_SPEC,
    describe,
    execution_model,
)
from benchmarks.e2e.registry import EXECUTION_MODEL, scenario_by_name
from benchmarks.e2e.runner import RunRequest, _resolve_run
from benchmarks.execution import affinity, provenance
from benchmarks.execution.affinity import CpuPinning, resolve_cpu_pinning
from benchmarks.execution.devices import parse_devices
from benchmarks.execution.environment import runtime_environment
from benchmarks.execution.paths import RuntimePaths
from benchmarks.execution.provenance import hardware_metadata


_METADATA = {
    "requested_gpu": "0",
    "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
    "torch_version": "test",
    "torchtitan_git_rev": "titan-rev",
    "benchmarks_git_rev": "bench-rev",
    "megatron_git_rev": "mcore-rev",
}


class ParseDevicesTests(unittest.TestCase):
    def test_one_device_is_still_a_one_entry_tuple(self) -> None:
        self.assertEqual(parse_devices("0"), ("0",))
        self.assertEqual(parse_devices("7"), ("7",))

    def test_a_comma_list_splits_in_order(self) -> None:
        self.assertEqual(parse_devices("0,1"), ("0", "1"))
        self.assertEqual(parse_devices("3,2,1,0"), ("3", "2", "1", "0"))

    def test_a_repeated_device_is_refused(self) -> None:
        # NGPU would claim more ranks than the run has devices.
        with self.assertRaisesRegex(ValueError, "more than once"):
            parse_devices("0,0")
        # Written two ways, still one device.
        with self.assertRaisesRegex(ValueError, "more than once"):
            parse_devices("0,00")

    def test_every_other_spelling_is_refused(self) -> None:
        """The string reaches CUDA_VISIBLE_DEVICES and nvidia-smi unchanged.

        Anything this parser reads differently than the driver would is a
        run measuring a device the manifest does not name.
        """
        for value in ("", "a", "0,", ",0", "0,,1", "0, 1", " 0", "-1", "0;1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_devices(value)


class KernelBenchDeviceTests(unittest.TestCase):
    def test_kernel_bench_refuses_more_than_one_device(self) -> None:
        result = CliRunner().invoke(cli, ["kernel-bench", "0,1"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("measures one device", result.output)

    def test_kernel_bench_refuses_a_malformed_device_list(self) -> None:
        result = CliRunner().invoke(cli, ["kernel-bench", "gpu0"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("comma-separated GPU indices", result.output)


class ExecutionOptionTests(unittest.TestCase):
    """The five new options, and the one thing they deliberately lack."""

    PARALLELISM_OPTIONS = (
        "--dp",
        "--pp",
        "--ep",
        "--pp-schedule",
        "--pp-microbatch-size",
    )

    def _parameters(self) -> dict:
        return {
            option: parameter
            for parameter in run_command.params
            for option in parameter.opts
        }

    def test_all_five_options_exist_on_both_execution_commands(self) -> None:
        parameters = self._parameters()
        for option in self.PARALLELISM_OPTIONS:
            with self.subTest(option=option):
                self.assertIn(option, parameters)

    def test_none_of_the_five_takes_an_environment_variable(self) -> None:
        """Each value must agree with the ``<gpu>`` positional.

        A positional has no environment form, so an exported ``PP=2`` would
        make a plain ``run 0 --scenario X`` fail its own world-size check
        with a message naming a flag the operator never passed. The three
        older axes have no such partner and keep their variables.
        """
        parameters = self._parameters()
        for option in self.PARALLELISM_OPTIONS:
            with self.subTest(option=option):
                self.assertIsNone(parameters[option].envvar)
        for option in ("--compile-mode", "--ac", "--model-size"):
            with self.subTest(option=option):
                self.assertIsNotNone(parameters[option].envvar)

    def test_the_option_block_records_why(self) -> None:
        self.assertIn("no environment variable", _execution_options.__doc__)


class RequestTests(unittest.TestCase):
    """What the CLI hands ``RunRequest``."""

    def _request(self, *arguments: str) -> RunRequest:
        seen = []

        def capture(request, **kwargs):
            seen.append(request)
            raise SystemExit(0)

        with mock.patch("benchmarks.cli.e2e.execute_run", side_effect=capture):
            CliRunner().invoke(cli, ["run", *arguments])
        self.assertEqual(len(seen), 1)
        return seen[0]

    def test_an_untouched_command_line_requests_no_parallelism(self) -> None:
        request = self._request("0", "--scenario", "piper1b_rope")
        self.assertIsNone(request.parallelism)

    def test_the_gpu_string_is_kept_exactly_as_typed(self) -> None:
        # Roughly one hundred manifests record it as requested_gpu, and
        # CUDA_VISIBLE_DEVICES is set from the same value.
        for value in ("0", "0,1", "3,2"):
            with self.subTest(value=value):
                request = self._request(value, "--scenario", "piper1b_rope")
                self.assertEqual(request.gpu, value)
                self.assertIsInstance(request.gpu, str)

    def test_the_five_options_build_one_spec(self) -> None:
        request = self._request(
            "0,1",
            "--scenario",
            "piper1b_rope",
            "--pp",
            "2",
            "--pp-schedule",
            "1F1B",
            "--pp-microbatch-size",
            "1",
        )
        self.assertEqual(
            request.parallelism,
            ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=1),
        )

    def test_an_option_left_out_takes_the_spec_default(self) -> None:
        request = self._request("0,1", "--scenario", "piper1b_rope", "--dp", "2")
        self.assertEqual(request.parallelism, ParallelismSpec(dp=2))

    def test_a_degree_below_one_is_refused_by_the_option(self) -> None:
        result = CliRunner().invoke(
            cli, ["run", "0", "--scenario", "piper1b_rope", "--pp", "0"]
        )
        self.assertNotEqual(result.exit_code, 0)

    def test_the_all_scenarios_sweep_gives_every_scenario_the_same_spec(
        self,
    ) -> None:
        """``_parallelism`` pops from a per-scenario copy, not from a shared dict.

        The sweep builds one request per scenario from ``{**options, ...}``.
        A pop that reached the caller's dict would leave the second scenario
        with no spec, and it would run the trivial one under a pp label.
        """
        seen = []

        def capture(request, **kwargs):
            seen.append(request)
            return mock.Mock(out_dir=Path("/tmp/out"))

        with mock.patch(
            "benchmarks.cli.e2e.execute_run", side_effect=capture
        ), mock.patch("benchmarks.cli.e2e._evaluate"), mock.patch(
            "benchmarks.cli.e2e.record_evaluation_status"
        ):
            result = CliRunner().invoke(
                cli, ["run-all", "0,1", "--all-scenarios", "--dp", "2"]
            )
        self.assertEqual(result.exit_code, 0, result.output)
        # Several scenarios, and every one of them carries the same spec.
        self.assertGreater(len(seen), 1)
        for request in seen:
            self.assertEqual(request.parallelism, ParallelismSpec(dp=2))
            self.assertEqual(request.gpu, "0,1")


class RuntimeEnvironmentTests(unittest.TestCase):
    def _environment(self, gpu: str, **kwargs) -> dict:
        paths = RuntimePaths.resolve(environment={"PATH": os.environ["PATH"]})
        return runtime_environment(
            paths, gpu, environment={"PATH": os.environ["PATH"]}, **kwargs
        )

    def test_one_device_is_unchanged(self) -> None:
        result = self._environment("0")
        self.assertEqual(result["NGPU"], "1")
        self.assertEqual(result["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(result["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        # Set only when a run has more than one rank to label, which is a
        # later stage; its absence is what a single-GPU log has always had.
        self.assertNotIn("LOG_RANK", result)

    def test_ngpu_follows_the_world_size(self) -> None:
        self.assertEqual(self._environment("0,1", world_size=2)["NGPU"], "2")
        self.assertEqual(
            self._environment("0,1", world_size=2)["CUDA_VISIBLE_DEVICES"], "0,1"
        )

    def test_a_world_size_below_one_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "world size"):
            self._environment("0", world_size=0)


class ProvenanceDeviceTests(unittest.TestCase):
    def _metadata(self, gpu: str, query: str):
        def fake_run_text(command, **kwargs):
            if command[0] == "nvidia-smi":
                return query
            return "stub"

        with mock.patch.object(provenance, "run_text", fake_run_text):
            return hardware_metadata(mock.Mock(), gpu, "auto")

    def test_one_device_records_the_line_and_the_label(self) -> None:
        label, metadata = self._metadata(
            "0", "0, NVIDIA H200, GPU-uuid, 570.211.01\n"
        )
        self.assertEqual(label, "nvidia-h200")
        self.assertEqual(metadata["requested_gpu"], "0")
        self.assertEqual(
            metadata["nvidia_smi"], "0, NVIDIA H200, GPU-uuid, 570.211.01"
        )

    def test_a_matching_device_set_records_every_line(self) -> None:
        label, metadata = self._metadata(
            "0,1",
            "0, NVIDIA H200, GPU-a, 570.211.01\n1, NVIDIA H200, GPU-b, 570.211.01\n",
        )
        self.assertEqual(label, "nvidia-h200")
        self.assertEqual(metadata["requested_gpu"], "0,1")
        self.assertEqual(len(metadata["nvidia_smi"].splitlines()), 2)

    def test_a_mixed_device_set_raises(self) -> None:
        """One run records one hardware label, so this is not one measurement."""
        with self.assertRaisesRegex(ValueError, "mixes GPU models"):
            self._metadata(
                "0,1",
                "0, NVIDIA H200, GPU-a, 570.211.01\n1, NVIDIA A100, GPU-b, 570.211.01\n",
            )

    def test_an_unavailable_query_still_does_not_fail_the_run(self) -> None:
        # Collecting provenance never fails a run; the mixed-model raise is
        # about the request, not about a failure to collect.
        label, metadata = self._metadata("0,1", "unavailable: no nvidia-smi")
        self.assertEqual(label, "gpu0-1")
        self.assertEqual(metadata["nvidia_smi"], "unavailable: no nvidia-smi")

    def test_a_degraded_query_at_one_device_does_not_raise(self) -> None:
        """A diagnostic is not a device roster.

        ``run_text`` merges stderr, so a failure can arrive as several lines
        that each hold a comma. Reading them as two devices would let a
        broken box raise where the old code recorded the string and went on.
        """
        label, metadata = self._metadata(
            "0",
            "unavailable: Command '['nvidia-smi', '--id=0']' failed\n"
            "second, line, with, commas",
        )
        self.assertIn("unavailable:", metadata["nvidia_smi"])
        self.assertTrue(label)

    def test_the_label_is_the_first_device_name(self) -> None:
        # Byte-identical to the pre-parallelism reading at one device: the
        # first comma of the query is the one after the first index.
        for gpu, query in (
            ("0", "0, NVIDIA H200, GPU-a, 570.211.01"),
            (
                "0,1",
                "0, NVIDIA H200, GPU-a, 570.211.01\n"
                "1, NVIDIA H200, GPU-b, 570.211.01",
            ),
        ):
            with self.subTest(gpu=gpu):
                self.assertEqual(self._metadata(gpu, query)[0], "nvidia-h200")


class AffinityDeviceTests(unittest.TestCase):
    def _sysfs(self, root: Path, device: str, node: str) -> None:
        path = root / "bus/pci/devices" / device
        path.mkdir(parents=True, exist_ok=True)
        (path / "numa_node").write_text(node)

    def _pinning(self, gpu: str, bus_ids: dict, root: Path) -> CpuPinning:
        def fake_run_text(command, **kwargs):
            index = command[1].removeprefix("--id=")
            return bus_ids[index]

        with mock.patch.object(
            affinity.shutil, "which", return_value="/usr/bin/numactl"
        ), mock.patch.object(affinity, "run_text", fake_run_text):
            return resolve_cpu_pinning(gpu, sysfs_root=root)

    def test_the_one_device_description_is_unchanged(self) -> None:
        """``--resume`` compares this string in every directory under out/."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._sysfs(root, "0000:1b:00.0", "1")
            pinning = self._pinning("0", {"0": "00000000:1B:00.0\n"}, root)
        self.assertEqual(
            pinning.description, "numactl --cpunodebind=1 --membind=1"
        )
        self.assertEqual(
            pinning.prefix, ("numactl", "--cpunodebind=1", "--membind=1")
        )

    def test_devices_on_one_node_pin_to_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._sysfs(root, "0000:1b:00.0", "1")
            self._sysfs(root, "0000:1c:00.0", "1")
            pinning = self._pinning(
                "0,1",
                {"0": "00000000:1B:00.0\n", "1": "00000000:1C:00.0\n"},
                root,
            )
        self.assertEqual(
            pinning.description, "numactl --cpunodebind=1 --membind=1"
        )

    def test_devices_that_span_nodes_run_unpinned_and_say_so(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._sysfs(root, "0000:1b:00.0", "0")
            self._sysfs(root, "0000:1c:00.0", "1")
            pinning = self._pinning(
                "0,1",
                {"0": "00000000:1B:00.0\n", "1": "00000000:1C:00.0\n"},
                root,
            )
        self.assertEqual(pinning.prefix, ())
        self.assertEqual(pinning.description, "none: devices 0,1 span NUMA nodes 0,1")

    def test_one_unresolvable_device_decides_the_whole_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._sysfs(root, "0000:1b:00.0", "0")
            pinning = self._pinning(
                "0,1", {"0": "00000000:1B:00.0\n", "1": "not-a-bus-id\n"}, root
            )
        self.assertEqual(pinning.prefix, ())
        self.assertEqual(
            pinning.description, "none: cannot resolve PCI bus id (not-a-bus-id)"
        )


class ManifestSchemaTenTests(unittest.TestCase):
    def _manifest(self, parallelism: ParallelismSpec) -> dict:
        scenario = scenario_by_name("piper1b_rope")
        return manifest_data(
            scenario,
            (scenario.arm("baseline"),),
            {"baseline": ["cmd"]},
            "test-gpu",
            _METADATA,
            (),
            "default",
            "sac",
            "1b",
            parallelism=parallelism,
        )

    def test_the_schema_is_eleven(self) -> None:
        self.assertEqual(MANIFEST_SCHEMA_VERSION, 12)
        self.assertEqual(self._manifest(TRIVIAL_SPEC)["schema_version"], 12)

    def test_the_trivial_spec_round_trips_through_json(self) -> None:
        recorded = json.loads(json.dumps(self._manifest(TRIVIAL_SPEC)))
        self.assertEqual(
            recorded["parallelism"], describe(TRIVIAL_SPEC, local_batch_size=4)
        )
        self.assertEqual(recorded["parallelism"]["world_size"], 1)
        self.assertEqual(recorded["parallelism"]["pp_schedule"], None)

    def test_a_pipelined_spec_round_trips_through_json(self) -> None:
        spec = ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=2)
        recorded = json.loads(json.dumps(self._manifest(spec)))
        self.assertEqual(
            recorded["parallelism"], describe(spec, local_batch_size=4)
        )
        self.assertEqual(recorded["parallelism"]["world_size"], 2)
        self.assertEqual(recorded["parallelism"]["n_microbatches"], 2)

    def test_the_default_parity_is_recorded_rather_than_left_out(self) -> None:
        """A key that appears only under ``shard`` would make a replicated
        run and a schema-11 run look the same, and one of the two states a
        fact the other cannot state."""
        recorded = json.loads(json.dumps(self._manifest(TRIVIAL_SPEC)))
        self.assertEqual(
            recorded["parallelism"]["dense_sharding"], "replicate"
        )

    def test_a_sharded_spec_round_trips_through_json(self) -> None:
        """Both halves reach the file: the parity the operator asked for,
        and the TorchTitan mesh it resolves to."""
        spec = ParallelismSpec(dp=2, dense_sharding="shard")
        recorded = json.loads(json.dumps(self._manifest(spec)))
        self.assertEqual(
            recorded["parallelism"], describe(spec, local_batch_size=4)
        )
        self.assertEqual(recorded["parallelism"]["dense_sharding"], "shard")
        self.assertEqual(recorded["parallelism"]["dp_replicate"], 1)
        self.assertEqual(recorded["parallelism"]["dp_shard"], 2)
        self.assertEqual(
            recorded["execution_model"], "2-gpu-plain-bf16-dp2-shard"
        )

    def test_an_omitted_parallelism_is_a_type_error(self) -> None:
        """A defaulted value would record dp 1 x pp 1 for any mesh."""
        scenario = scenario_by_name("piper1b_rope")
        with self.assertRaises(TypeError):
            manifest_data(
                scenario,
                (scenario.arm("baseline"),),
                {"baseline": ["cmd"]},
                "test-gpu",
                _METADATA,
                (),
                "default",
                "sac",
                "1b",
            )


class ExecutionModelFollowsTheMeshTests(unittest.TestCase):
    """The manifest describes the run it recorded, not a constant.

    A manifest exists so a directory self-describes without a git-rev
    lookup. One constant cannot describe two meshes, so the field is
    composed from the run's own spec -- and the trivial answer has to be the
    string every directory since schema 7 already carries.
    """

    def _manifest(self, parallelism: ParallelismSpec) -> dict:
        scenario = scenario_by_name("piper1b_rope")
        return manifest_data(
            scenario,
            (scenario.arm("baseline"),),
            {"baseline": ["cmd"]},
            "test-gpu",
            _METADATA,
            (),
            "default",
            "sac",
            "1b",
            parallelism=parallelism,
        )

    def test_the_trivial_spec_records_the_string_it_always_recorded(self) -> None:
        self.assertEqual(
            self._manifest(TRIVIAL_SPEC)["execution_model"],
            "single-gpu-plain-bf16-no-fsdp",
        )
        self.assertEqual(
            self._manifest(TRIVIAL_SPEC)["execution_model"], EXECUTION_MODEL
        )

    def test_a_pipelined_spec_records_its_own_mesh(self) -> None:
        spec = ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=2)
        self.assertEqual(
            self._manifest(spec)["execution_model"],
            "2-gpu-plain-bf16-no-fsdp-pp2-1F1B",
        )

    def test_the_field_is_whatever_the_spec_module_composes(self) -> None:
        # One derivation, so the manifest cannot drift from the module that
        # owns the vocabulary.
        for spec in (
            TRIVIAL_SPEC,
            ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=2),
        ):
            self.assertEqual(
                self._manifest(spec)["execution_model"], execution_model(spec)
            )


class ExecutionModelIsNotResumeGatedTests(unittest.TestCase):
    """It is derived from ``parallelism``, which the resume already gates.

    Gating it too would refuse the same run twice and report the derived
    field rather than the field an operator set.
    """

    def setUp(self) -> None:
        self.scenario = scenario_by_name("piper1b_rope")
        self.arms = (self.scenario.arm("baseline"),)

    def test_a_manifest_whose_only_difference_is_the_derived_field_resumes(
        self,
    ) -> None:
        manifest = manifest_data(
            self.scenario,
            self.arms,
            {"baseline": ["cmd"]},
            "test-gpu",
            _METADATA,
            (),
            "default",
            "sac",
            "1b",
            parallelism=TRIVIAL_SPEC,
        )
        manifest["execution_model"] = "something-else-entirely"
        self.assertEqual(
            _resume_mismatches(
                manifest,
                self.scenario,
                self.arms,
                "test-gpu",
                _METADATA,
                (),
                "default",
                "sac",
                "1b",
                parallelism=TRIVIAL_SPEC,
            ),
            [],
        )

    def test_the_spec_it_derives_from_is_gated(self) -> None:
        manifest = manifest_data(
            self.scenario,
            self.arms,
            {"baseline": ["cmd"]},
            "test-gpu",
            _METADATA,
            (),
            "default",
            "sac",
            "1b",
            parallelism=TRIVIAL_SPEC,
        )
        self.assertIn(
            "parallelism",
            _resume_mismatches(
                manifest,
                self.scenario,
                self.arms,
                "test-gpu",
                _METADATA,
                (),
                "default",
                "sac",
                "1b",
                parallelism=ParallelismSpec(pp=2, pp_schedule="1F1B"),
            ),
        )


class ResumeParallelismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = scenario_by_name("piper1b_rope")
        self.arms = (self.scenario.arm("baseline"),)

    def _manifest(self, parallelism: ParallelismSpec) -> dict:
        return manifest_data(
            self.scenario,
            self.arms,
            {"baseline": ["cmd"]},
            "test-gpu",
            _METADATA,
            (),
            "default",
            "sac",
            "1b",
            parallelism=parallelism,
        )

    def _mismatches(self, manifest: dict, parallelism: ParallelismSpec):
        return _resume_mismatches(
            manifest,
            self.scenario,
            self.arms,
            "test-gpu",
            _METADATA,
            (),
            "default",
            "sac",
            "1b",
            parallelism=parallelism,
        )

    def test_a_schema_nine_directory_still_resumes_as_single_gpu(self) -> None:
        """Every directory under out/ predates the axis and carries no key."""
        manifest = self._manifest(TRIVIAL_SPEC)
        del manifest["parallelism"]
        manifest["schema_version"] = 9
        self.assertEqual(self._mismatches(manifest, TRIVIAL_SPEC), [])

    def test_a_schema_nine_directory_refuses_a_parallel_request(self) -> None:
        manifest = self._manifest(TRIVIAL_SPEC)
        del manifest["parallelism"]
        manifest["schema_version"] = 9
        self.assertIn(
            "parallelism",
            self._mismatches(
                manifest, ParallelismSpec(pp=2, pp_schedule="1F1B")
            ),
        )

    def test_the_same_spec_resumes_and_a_different_one_does_not(self) -> None:
        spec = ParallelismSpec(pp=2, pp_schedule="1F1B")
        manifest = self._manifest(spec)
        self.assertEqual(self._mismatches(manifest, spec), [])
        self.assertIn("parallelism", self._mismatches(manifest, TRIVIAL_SPEC))
        self.assertIn(
            "parallelism",
            self._mismatches(
                manifest, ParallelismSpec(pp=2, pp_schedule="Interleaved1F1B")
            ),
        )

    def test_the_microbatch_size_alone_refuses_a_resume(self) -> None:
        # It decides every arm's command line, and --resume compares no
        # command line, so the record is what has to carry it.
        recorded = ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=1)
        requested = ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=2)
        self.assertIn(
            "parallelism", self._mismatches(self._manifest(recorded), requested)
        )

    def test_the_dense_sharding_value_alone_refuses_a_resume(self) -> None:
        """It is a comparability boundary: the two parities hold different
        amounts of optimizer state per rank and exchange different tensors.
        ``_resume_mismatches`` compares the whole record, so the key is gated
        the moment ``describe`` records it.
        """
        recorded = ParallelismSpec(dp=2)
        requested = ParallelismSpec(dp=2, dense_sharding="shard")
        self.assertIn(
            "parallelism", self._mismatches(self._manifest(recorded), requested)
        )
        self.assertIn(
            "parallelism", self._mismatches(self._manifest(requested), recorded)
        )

    def test_a_schema_eleven_block_cannot_claim_the_default_parity(self) -> None:
        """A schema-11 ``parallelism`` block predates the key.

        Reading its absence as ``replicate`` would be an inference. Every
        such run really was replicated, but the block cannot say so, and the
        safe direction is to refuse the resume rather than to record a parity
        the file never carried. (A resume across this commit is refused by
        ``benchmarks_git_rev`` anyway; this pins which way the record itself
        reads.)
        """
        manifest = self._manifest(TRIVIAL_SPEC)
        manifest["schema_version"] = 11
        del manifest["parallelism"]["dense_sharding"]
        self.assertIn("parallelism", self._mismatches(manifest, TRIVIAL_SPEC))


class ResolveRunTests(unittest.TestCase):
    """``_resolve_run`` resolves the mesh, and derives the regions from it."""

    def _resolve(self, **kwargs):
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", dict(_METADATA)),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            return _resolve_run(
                RunRequest(scenario_name="piper1b_rope", **kwargs),
                {"PATH": os.environ["PATH"]},
            )

    def test_a_single_gpu_run_resolves_to_the_trivial_spec(self) -> None:
        resolved = self._resolve(gpu="0")
        self.assertEqual(resolved[10], TRIVIAL_SPEC)

    def test_a_named_trivial_spec_resolves_the_same_way(self) -> None:
        self.assertEqual(self._resolve(gpu="0", parallelism=TRIVIAL_SPEC)[10], TRIVIAL_SPEC)

    def test_a_mesh_that_does_not_fill_the_device_list_is_refused(self) -> None:
        # Rule 1: not "at most". An under-filled request would leave a GPU
        # idle and publish the number under the whole device list.
        with self.assertRaisesRegex(ValueError, "does not match"):
            self._resolve(gpu="0,1")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self._resolve(gpu="0", parallelism=ParallelismSpec(pp=2, pp_schedule="1F1B"))

    def test_a_legal_mesh_resolves(self) -> None:
        spec = ParallelismSpec(pp=2, pp_schedule="1F1B")
        self.assertEqual(self._resolve(gpu="0,1", parallelism=spec)[10], spec)

    def test_a_single_gpu_run_still_declares_its_block_regions(self) -> None:
        scenario = self._resolve(gpu="0")[1]
        self.assertEqual(
            sorted(region.name for region in scenario.regions),
            ["backward_block", "forward_block"],
        )
        self.assertEqual(
            [region.invocations_per_window for region in scenario.regions],
            [80, 80],
        )

    def test_a_pipelined_run_declares_no_regions(self) -> None:
        """No rank holds every block, so the declared count is unreachable.

        ``piper_block_regions`` asks for ``n_layers * profiler_active``
        invocations per window, and that count IS the region's identity. A
        rank of a two-stage pipeline holds half the layers and runs each of
        them once per microbatch, so it never reaches 80. Deriving a
        per-rank count instead would be rule 7 rewritten rather than applied.
        Real pipelined traces exist, but no trace analysis has established a
        unique per-rank invocation identity that could replace this rule.
        """
        scenario = self._resolve(
            gpu="0,1", parallelism=ParallelismSpec(pp=2, pp_schedule="1F1B")
        )[1]
        self.assertEqual(scenario.regions, ())

    def test_the_scenario_that_declares_none_is_unaffected(self) -> None:
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", dict(_METADATA)),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            scenario = _resolve_run(
                RunRequest(
                    gpu="0", scenario_name="piper1b_megatron", ac_mode="none"
                ),
                {"PATH": os.environ["PATH"]},
            )[1]
        self.assertEqual(scenario.regions, ())

    def test_a_malformed_device_list_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "comma-separated GPU indices"):
            self._resolve(gpu="gpu0")

    def test_an_illegal_mesh_is_refused_before_any_host_probe(self) -> None:
        def never(*args, **kwargs):
            raise AssertionError("a host probe ran for a refused mesh")

        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata", side_effect=never
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning", side_effect=never
        ):
            with self.assertRaises(ValueError):
                _resolve_run(
                    RunRequest(gpu="0,1", scenario_name="piper1b_rope"),
                    {"PATH": os.environ["PATH"]},
                )


if __name__ == "__main__":
    unittest.main()
