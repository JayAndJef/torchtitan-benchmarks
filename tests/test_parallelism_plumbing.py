"""The parallelism axis threaded through the harness, without leaving one GPU.

``tests/test_parallelism.py`` covers the axis itself -- the degrees, the
schedules and the sixteen validator rules. This module covers the path the
value takes: the ``<gpu>`` positional read as a device set, the six CLI
options, ``RunRequest``, ``_resolve_run``, the child environment, the
provenance query, the NUMA walk, and manifest schema 16.

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
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from click.testing import CliRunner

from benchmarks.artifacts.manifests import (
    MANIFEST_SCHEMA_VERSION,
    _resume_mismatches,
    manifest_data,
)
from benchmarks.cli.e2e import (
    _PARALLELISM_OPTIONS,
    _execution_options,
    run_command,
)
from benchmarks.cli.main import cli
from benchmarks.e2e.parallelism import (
    ParallelismSpec,
    TRIVIAL_SPEC,
    describe,
    execution_model,
)
from benchmarks.e2e.registry import EXECUTION_MODEL, scenario_by_name
from benchmarks.e2e.runner import RunRequest, _resolve_run, execute_run
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
    """The six parallelism options, and the one thing they deliberately
    lack."""

    PARALLELISM_OPTIONS = (
        "--dp",
        "--pp",
        "--ep",
        "--pp-schedule",
        "--pp-microbatch-size",
        "--dense-sharding",
    )

    def _parameters(self) -> dict:
        return {
            option: parameter
            for parameter in run_command.params
            for option in parameter.opts
        }

    def test_all_six_options_exist_on_both_execution_commands(self) -> None:
        parameters = self._parameters()
        for option in self.PARALLELISM_OPTIONS:
            with self.subTest(option=option):
                self.assertIn(option, parameters)

    def test_the_option_roster_is_the_one_the_spec_is_built_from(self) -> None:
        """``_parallelism`` pops exactly these six keywords, so an option
        added here and not there would be dropped, and one added there and
        not here would build a spec from a value no flag can set."""
        self.assertEqual(
            tuple(name for name, _ in _PARALLELISM_OPTIONS),
            ("dp", "pp", "ep", "pp_schedule", "pp_microbatch_size",
             "dense_sharding"),
        )
        self.assertEqual(
            len(_PARALLELISM_OPTIONS), len(self.PARALLELISM_OPTIONS)
        )

    def test_the_recorded_defaults_are_the_specs_own(self) -> None:
        """A default written out twice can drift, and the drift would build
        a spec the operator did not ask for."""
        defaults = dict(_PARALLELISM_OPTIONS)
        trivial = ParallelismSpec()
        for name, default in defaults.items():
            with self.subTest(option=name):
                self.assertEqual(getattr(trivial, name), default)

    def test_none_of_the_six_takes_an_environment_variable(self) -> None:
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
        request = self._request("0", "--scenario", "piper1b_megatron")
        self.assertIsNone(request.parallelism)

    def test_the_gpu_string_is_kept_exactly_as_typed(self) -> None:
        # Roughly one hundred manifests record it as requested_gpu, and
        # CUDA_VISIBLE_DEVICES is set from the same value.
        for value in ("0", "0,1", "3,2"):
            with self.subTest(value=value):
                request = self._request(value, "--scenario", "piper1b_megatron")
                self.assertEqual(request.gpu, value)
                self.assertIsInstance(request.gpu, str)

    def test_the_dense_sharding_option_reaches_the_spec(self) -> None:
        request = self._request(
            "0,1",
            "--scenario",
            "piper1b_megatron",
            "--dp",
            "2",
            "--dense-sharding",
            "zero3",
        )
        self.assertEqual(
            request.parallelism,
            ParallelismSpec(dp=2, dense_sharding="zero3"),
        )

    def test_the_dense_sharding_option_refuses_an_undeclared_value(
        self,
    ) -> None:
        """**Click refuses it, and the exit code is what says so.**

        A nonzero exit proves nothing here: a legal ``--dense-sharding
        zero3`` also exits nonzero, because the run then starts and fails on
        this host for its own reasons. Click's usage error is exit 2, and
        it names the roster. Without the ``click.Choice`` the string would
        reach ``ParallelismSpec.__post_init__``, raise, and exit 1 -- a
        refusal in the right direction under the wrong code, which this
        assertion separates.

        **The value under test is ``shard``, which is the RETIRED
        spelling.** Three recorded cells carry it, so an operator who reads
        an old manifest can type it. It must reach the roster message rather
        than the new ``zero3`` behaviour.
        """
        result = CliRunner().invoke(
            cli,
            [
                "run",
                "0,1",
                "--scenario",
                "piper1b_megatron",
                "--dp",
                "2",
                "--dense-sharding",
                "shard",
            ],
        )
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("'replicate', 'zero1', 'zero3'", result.output)

    def test_the_pipeline_options_build_one_spec(self) -> None:
        request = self._request(
            "0,1",
            "--scenario",
            "piper1b_megatron",
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
        request = self._request("0,1", "--scenario", "piper1b_megatron", "--dp", "2")
        self.assertEqual(request.parallelism, ParallelismSpec(dp=2))

    def test_a_degree_below_one_is_refused_by_the_option(self) -> None:
        result = CliRunner().invoke(
            cli, ["run", "0", "--scenario", "piper1b_megatron", "--pp", "0"]
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
                cli,
                [
                    "run-all",
                    "0,1",
                    "--all-scenarios",
                    "--ac",
                    "none",
                    "--dp",
                    "2",
                ],
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


class ManifestSchemaSixteenTests(unittest.TestCase):
    def _manifest(
        self,
        parallelism: ParallelismSpec,
        megatron_p2p_sync: str = "on",
        megatron_nan_guard: str = "on",
        megatron_precision: str = "stock",
    ) -> dict:
        scenario = scenario_by_name("piper1b_megatron")
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
            megatron_p2p_sync=megatron_p2p_sync,
            megatron_nan_guard=megatron_nan_guard,
            megatron_precision=megatron_precision,
        )

    def test_the_schema_is_sixteen(self) -> None:
        self.assertEqual(MANIFEST_SCHEMA_VERSION, 16)
        self.assertEqual(self._manifest(TRIVIAL_SPEC)["schema_version"], 16)

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
        spec = ParallelismSpec(dp=2, dense_sharding="zero3")
        recorded = json.loads(json.dumps(self._manifest(spec)))
        self.assertEqual(
            recorded["parallelism"], describe(spec, local_batch_size=4)
        )
        self.assertEqual(recorded["parallelism"]["dense_sharding"], "zero3")
        self.assertEqual(recorded["parallelism"]["dp_replicate"], 1)
        self.assertEqual(recorded["parallelism"]["dp_shard"], 2)
        self.assertEqual(
            recorded["execution_model"], "2-gpu-plain-bf16-dp2-zero3"
        )

    def test_an_omitted_parallelism_is_a_type_error(self) -> None:
        """A defaulted value would record dp 1 x pp 1 for any mesh."""
        scenario = scenario_by_name("piper1b_megatron")
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

    def test_the_p2p_sync_default_is_recorded_rather_than_left_out(
        self,
    ) -> None:
        """A key that appears only under ``off`` would make a schema-13
        run at ``on`` and a schema-12 run look the same, and one of the two
        states a fact the other cannot state."""
        recorded = json.loads(json.dumps(self._manifest(TRIVIAL_SPEC)))
        self.assertEqual(recorded["megatron_p2p_sync"], "on")
        # Its own field beside compile_mode, not a key of the parallelism
        # block: it is a treatment of the pipeline messages, not a degree.
        self.assertNotIn("megatron_p2p_sync", recorded["parallelism"])

    def test_the_p2p_sync_value_round_trips_through_json(self) -> None:
        spec = ParallelismSpec(pp=2, pp_schedule="1F1B")
        recorded = json.loads(json.dumps(self._manifest(spec, "off")))
        self.assertEqual(recorded["megatron_p2p_sync"], "off")
        self.assertEqual(
            recorded["parallelism"], describe(spec, local_batch_size=4)
        )
        # The value is not part of the execution model, so two runs of one
        # mesh under the two values record the same string there.
        self.assertEqual(
            recorded["execution_model"],
            self._manifest(spec, "on")["execution_model"],
        )

    def test_an_omitted_p2p_sync_is_a_type_error(self) -> None:
        """A writer that defaulted it would record ``on`` for a run that
        turned the sync off, and the two are a comparability boundary."""
        scenario = scenario_by_name("piper1b_megatron")
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
                parallelism=TRIVIAL_SPEC,
            )

    def test_the_nan_guard_default_is_recorded_rather_than_left_out(
        self,
    ) -> None:
        """A key that appears only under ``off`` would make a schema-14 run
        at ``on`` and a schema-13 run look the same."""
        recorded = json.loads(json.dumps(self._manifest(TRIVIAL_SPEC)))
        self.assertEqual(recorded["megatron_nan_guard"], "on")
        # Its own field, not a key of the parallelism block: a treatment
        # of the stock engine's checks, not a degree.
        self.assertNotIn("megatron_nan_guard", recorded["parallelism"])

    def test_the_nan_guard_value_round_trips_through_json(self) -> None:
        recorded = json.loads(
            json.dumps(self._manifest(TRIVIAL_SPEC, megatron_nan_guard="off"))
        )
        self.assertEqual(recorded["megatron_nan_guard"], "off")
        self.assertEqual(recorded["megatron_p2p_sync"], "on")
        # Not part of the execution model, so two runs of one mesh under
        # the two values record the same string there.
        self.assertEqual(
            recorded["execution_model"],
            self._manifest(TRIVIAL_SPEC)["execution_model"],
        )

    def test_the_precision_value_round_trips_through_json(self) -> None:
        recorded = json.loads(
            json.dumps(
                self._manifest(TRIVIAL_SPEC, megatron_precision="lean")
            )
        )
        self.assertEqual(recorded["megatron_precision"], "lean")
        # Not part of the execution model either, for the same reason: it
        # is a treatment of the optimizer state and not a degree.
        self.assertEqual(
            recorded["execution_model"],
            self._manifest(TRIVIAL_SPEC)["execution_model"],
        )

    def test_the_precision_default_is_recorded_rather_than_left_out(
        self,
    ) -> None:
        """An absent key would read as ``stock`` by inference. The record
        is what separates "this run held 18 bytes for each parameter" from
        "this file predates the question"."""
        self.assertEqual(
            self._manifest(TRIVIAL_SPEC)["megatron_precision"], "stock"
        )

    def test_an_omitted_nan_guard_is_a_type_error(self) -> None:
        """A writer that defaulted it would record ``on`` for a run that
        turned the guard off, and the two are a comparability boundary."""
        scenario = scenario_by_name("piper1b_megatron")
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
                parallelism=TRIVIAL_SPEC,
                megatron_p2p_sync="on",
            )

    def test_an_omitted_precision_is_a_type_error(self) -> None:
        """A writer that defaulted it would record ``stock`` for a run that
        held 10 bytes for each parameter rather than 18."""
        scenario = scenario_by_name("piper1b_megatron")
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
                parallelism=TRIVIAL_SPEC,
                megatron_p2p_sync="on",
                megatron_nan_guard="on",
            )


class ExecutionModelFollowsTheMeshTests(unittest.TestCase):
    """The manifest describes the run it recorded, not a constant.

    A manifest exists so a directory self-describes without a git-rev
    lookup. One constant cannot describe two meshes, so the field is
    composed from the run's own spec -- and the trivial answer has to be the
    string every directory since schema 7 already carries.
    """

    def _manifest(self, parallelism: ParallelismSpec) -> dict:
        scenario = scenario_by_name("piper1b_megatron")
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
            megatron_p2p_sync="on",
            megatron_nan_guard="on",
            megatron_precision="stock",
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
        self.scenario = scenario_by_name("piper1b_megatron")
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
            megatron_p2p_sync="on",
            megatron_nan_guard="on",
            megatron_precision="stock",
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
                megatron_p2p_sync="on",
                megatron_nan_guard="on",
                megatron_precision="stock",
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
            megatron_p2p_sync="on",
            megatron_nan_guard="on",
            megatron_precision="stock",
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
                megatron_p2p_sync="on",
                megatron_nan_guard="on",
                megatron_precision="stock",
            ),
        )


class ResumeParallelismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = scenario_by_name("piper1b_megatron")
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
            megatron_p2p_sync="on",
            megatron_nan_guard="on",
            megatron_precision="stock",
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
            megatron_p2p_sync="on",
            megatron_nan_guard="on",
            megatron_precision="stock",
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
        requested = ParallelismSpec(dp=2, dense_sharding="zero3")
        self.assertIn(
            "parallelism", self._mismatches(self._manifest(recorded), requested)
        )
        self.assertIn(
            "parallelism", self._mismatches(self._manifest(requested), recorded)
        )

    def test_a_block_without_the_key_cannot_claim_the_default_parity(
        self,
    ) -> None:
        """A schema-11 ``parallelism`` block predates the key.

        Reading its absence as ``replicate`` would be an inference. Every
        such run really was replicated, but the block cannot say so, and the
        safe direction is to refuse the resume rather than to record a parity
        the file never carried.

        **The comparison reads no ``schema_version``.** ``_resume_mismatches``
        compares the whole ``parallelism`` block, so the missing key alone is
        what refuses this. Setting a version here would suggest a gate that
        does not exist. (A resume across this commit is refused by
        ``benchmarks_git_rev`` anyway; this pins which way the record itself
        reads.)
        """
        manifest = self._manifest(TRIVIAL_SPEC)
        del manifest["parallelism"]["dense_sharding"]
        self.assertIn("parallelism", self._mismatches(manifest, TRIVIAL_SPEC))


class ResumeMegatronP2pSyncTests(unittest.TestCase):
    """``--resume`` gates ``megatron_p2p_sync`` the way it gates the
    compile mode: the same value resumes, a different one is refused in
    either direction, and a directory that predates the field reads as
    ``on``.
    """

    PP2 = ParallelismSpec(pp=2, pp_schedule="1F1B")

    def setUp(self) -> None:
        self.scenario = scenario_by_name("piper1b_megatron")
        self.arms = (self.scenario.arm("baseline"),)

    def _manifest(self, megatron_p2p_sync: str) -> dict:
        return manifest_data(
            self.scenario,
            self.arms,
            {"baseline": ["cmd"]},
            "test-gpu",
            _METADATA,
            (),
            "default",
            "none",
            "1b",
            parallelism=self.PP2,
            megatron_p2p_sync=megatron_p2p_sync,
            megatron_nan_guard="on",
            megatron_precision="stock",
        )

    def _mismatches(self, manifest: dict, megatron_p2p_sync: str) -> list[str]:
        return _resume_mismatches(
            manifest,
            self.scenario,
            self.arms,
            "test-gpu",
            _METADATA,
            (),
            "default",
            "none",
            "1b",
            parallelism=self.PP2,
            megatron_p2p_sync=megatron_p2p_sync,
            megatron_nan_guard="on",
            megatron_precision="stock",
        )

    def test_the_same_value_resumes_and_a_different_one_is_refused(
        self,
    ) -> None:
        for recorded, requested in (("on", "off"), ("off", "on")):
            with self.subTest(recorded=recorded, requested=requested):
                manifest = self._manifest(recorded)
                self.assertEqual(self._mismatches(manifest, recorded), [])
                self.assertIn(
                    "megatron_p2p_sync", self._mismatches(manifest, requested)
                )

    def test_the_value_alone_refuses_a_resume(self) -> None:
        """Toggling only the value toggles only that mismatch, so the
        refusal names the field rather than something that co-varies with
        it -- and it is not reported as a parallelism mismatch."""
        manifest = self._manifest("off")
        refused = self._mismatches(manifest, "on")
        self.assertEqual(refused, ["megatron_p2p_sync"])
        self.assertNotIn("parallelism", refused)

    def test_a_schema_twelve_directory_reads_as_on(self) -> None:
        """No run before schema 13 could turn the sync off, so the absent
        key is a record of ``on`` and not an inference: the run kept stock
        Megatron's own synchronize. A request for ``off`` against such a
        directory is refused rather than silently changing the treatment."""
        manifest = self._manifest("on")
        del manifest["megatron_p2p_sync"]
        manifest["schema_version"] = 12
        self.assertEqual(self._mismatches(manifest, "on"), [])
        self.assertIn("megatron_p2p_sync", self._mismatches(manifest, "off"))


class ResumeMegatronNanGuardTests(unittest.TestCase):
    """``--resume`` gates ``megatron_nan_guard`` the way it gates the p2p
    value: the same value resumes, a different one is refused in either
    direction, and a directory that predates the field reads as ``on``.
    """

    def setUp(self) -> None:
        self.scenario = scenario_by_name("piper_megatron_stock")
        self.arms = (self.scenario.arm("baseline"),)

    def _manifest(
        self, megatron_nan_guard: str, megatron_precision: str = "stock"
    ) -> dict:
        return manifest_data(
            self.scenario,
            self.arms,
            {"baseline": ["cmd"]},
            "test-gpu",
            _METADATA,
            (),
            "default",
            "none",
            "1b",
            parallelism=TRIVIAL_SPEC,
            megatron_p2p_sync="on",
            megatron_nan_guard=megatron_nan_guard,
            megatron_precision=megatron_precision,
        )

    def _mismatches(
        self,
        manifest: dict,
        megatron_nan_guard: str,
        megatron_precision: str = "stock",
    ) -> list[str]:
        return _resume_mismatches(
            manifest,
            self.scenario,
            self.arms,
            "test-gpu",
            _METADATA,
            (),
            "default",
            "none",
            "1b",
            parallelism=TRIVIAL_SPEC,
            megatron_p2p_sync="on",
            megatron_nan_guard=megatron_nan_guard,
            megatron_precision=megatron_precision,
        )

    def test_the_same_value_resumes_and_a_different_one_is_refused(
        self,
    ) -> None:
        for recorded, requested in (("on", "off"), ("off", "on")):
            with self.subTest(recorded=recorded, requested=requested):
                manifest = self._manifest(recorded)
                self.assertEqual(self._mismatches(manifest, recorded), [])
                self.assertIn(
                    "megatron_nan_guard", self._mismatches(manifest, requested)
                )

    def test_the_value_alone_refuses_a_resume(self) -> None:
        """Toggling only the value toggles only that mismatch, so the
        refusal names the field and not something that co-varies with it."""
        manifest = self._manifest("off")
        refused = self._mismatches(manifest, "on")
        self.assertEqual(refused, ["megatron_nan_guard"])

    def test_a_schema_thirteen_directory_reads_as_on(self) -> None:
        """No run before schema 14 could turn the guard off through the
        harness, so the absent key is a record of ``on``. A request for
        ``off`` against such a directory is refused rather than silently
        changing the treatment under the recorded label."""
        manifest = self._manifest("on")
        del manifest["megatron_nan_guard"]
        manifest["schema_version"] = 13
        self.assertEqual(self._mismatches(manifest, "on"), [])
        self.assertIn("megatron_nan_guard", self._mismatches(manifest, "off"))


class ResumeMegatronPrecisionTests(unittest.TestCase):
    """``--resume`` gates ``megatron_precision`` the way it gates the two
    values above: the same value resumes, a different one is refused in
    either direction, and a directory that predates the field reads as
    ``stock``.

    It also carries the schema-15 rename. A manifest that records the
    retired ``shard`` spelling is refused with a message that names the
    rename, rather than a bare ``parallelism`` key.
    """

    def setUp(self) -> None:
        self.scenario = scenario_by_name("piper_megatron_stock")
        self.arms = (self.scenario.arm("baseline"),)

    def _manifest(
        self,
        megatron_precision: str,
        parallelism: ParallelismSpec = TRIVIAL_SPEC,
    ) -> dict:
        return manifest_data(
            self.scenario,
            self.arms,
            {"baseline": ["cmd"]},
            "test-gpu",
            _METADATA,
            (),
            "default",
            "none",
            "1b",
            parallelism=parallelism,
            megatron_p2p_sync="on",
            megatron_nan_guard="on",
            megatron_precision=megatron_precision,
        )

    def _mismatches(
        self,
        manifest: dict,
        megatron_precision: str,
        parallelism: ParallelismSpec = TRIVIAL_SPEC,
    ) -> list[str]:
        return _resume_mismatches(
            manifest,
            self.scenario,
            self.arms,
            "test-gpu",
            _METADATA,
            (),
            "default",
            "none",
            "1b",
            parallelism=parallelism,
            megatron_p2p_sync="on",
            megatron_nan_guard="on",
            megatron_precision=megatron_precision,
        )

    def test_the_same_value_resumes_and_a_different_one_is_refused(
        self,
    ) -> None:
        for recorded, requested in (("stock", "lean"), ("lean", "stock")):
            with self.subTest(recorded=recorded, requested=requested):
                manifest = self._manifest(recorded)
                self.assertEqual(self._mismatches(manifest, recorded), [])
                self.assertIn(
                    "megatron_precision",
                    self._mismatches(manifest, requested),
                )

    def test_a_schema_fifteen_directory_reads_as_stock(self) -> None:
        """No run before schema 16 could ask for the lean recipe, so the
        absent key is a record of ``stock`` and not an inference."""
        manifest = self._manifest("stock")
        del manifest["megatron_precision"]
        manifest["schema_version"] = 15
        self.assertEqual(self._mismatches(manifest, "stock"), [])
        self.assertIn(
            "megatron_precision", self._mismatches(manifest, "lean")
        )

    def test_a_retired_shard_record_names_the_rename(self) -> None:
        """The run really held the ZeRO-3 parity, so the rename takes no
        number away. This gate compares two vocabularies, and a bare
        ``parallelism`` would send the operator looking for a degree that
        did not move."""
        from benchmarks.artifacts.manifests import RETIRED_DENSE_SHARDING

        spec = ParallelismSpec(dp=2, dense_sharding="zero3")
        manifest = self._manifest("stock", parallelism=spec)
        manifest["parallelism"]["dense_sharding"] = RETIRED_DENSE_SHARDING
        manifest["schema_version"] = 14
        refused = self._mismatches(manifest, "stock", parallelism=spec)
        self.assertEqual(len(refused), 1)
        self.assertTrue(refused[0].startswith("parallelism"))
        self.assertIn("retired spelling of 'zero3'", refused[0])

    def test_a_current_zero3_record_resumes(self) -> None:
        """The rename message is for the retired spelling alone."""
        spec = ParallelismSpec(dp=2, dense_sharding="zero3")
        manifest = self._manifest("stock", parallelism=spec)
        self.assertEqual(
            self._mismatches(manifest, "stock", parallelism=spec), []
        )


class ResolveRunTests(unittest.TestCase):
    """``_resolve_run`` resolves the mesh, and derives the regions from it."""

    def _resolve(self, scenario_name: str = "piper1b_megatron", **kwargs):
        kwargs.setdefault("ac_mode", "none")
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", dict(_METADATA)),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            return _resolve_run(
                RunRequest(scenario_name=scenario_name, **kwargs),
                {"PATH": os.environ["PATH"]},
            )

    def test_an_arm_subset_narrows_the_engine_set_spec_rule_16_reads(
        self,
    ) -> None:
        """**The repair rule 16's messages name, checked where it happens.**

        ``_resolve_run`` builds the ``engines`` argument from the arms this
        run will really start, so ``run --arm NAME`` narrows it. Rule 16
        refuses the sharded parity to the tuned megatron driver, and its
        messages tell the operator to select the TorchTitan arms alone. That
        advice is only true if the selector reaches the engine set, and this
        is the one test that says it does.

        ``piper1b_megatron`` is the scenario that holds both engines. Its
        ``baseline`` arm is the tuned megatron driver and ``titan_stock`` is
        a TorchTitan arm. The scenario declines ``--ac sac``, so both cases
        pass ``ac_mode="none"``.

        **It says the rule admits the run. It does not say the run
        succeeds.** ``parallelize_piper1b`` now admits an explicit shard
        degree, so the subprocess no longer refuses the selected arm -- but
        no sharded arm has run on a GPU, so the rule is all this checks.
        """
        sharded = ParallelismSpec(dp=2, dense_sharding="zero3")
        with self.assertRaisesRegex(
            ValueError, r"--dense-sharding zero3 is not implemented"
        ):
            self._resolve(
                scenario_name="piper1b_megatron",
                gpu="0,1",
                ac_mode="none",
                parallelism=sharded,
            )
        resolved = self._resolve(
            scenario_name="piper1b_megatron",
            gpu="0,1",
            ac_mode="none",
            arm_names=("titan_stock",),
            parallelism=sharded,
        )
        self.assertEqual(
            [arm.name for arm in resolved[2]], ["titan_stock"]
        )
        self.assertEqual(
            {arm.launcher for arm in resolved[2]}, {"torchtitan"}
        )
        self.assertEqual(resolved[10], sharded)

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
                    RunRequest(gpu="0,1", scenario_name="piper1b_megatron"),
                    {"PATH": os.environ["PATH"]},
                )


class ConnectionLimitPreconditionTests(unittest.TestCase):
    """The one host variable that stops a sharded Megatron run at parsing.

    ``megatron/training/arguments.py`` asserts
    ``CUDA_DEVICE_MAX_CONNECTIONS != "1"`` under ``--use-megatron-fsdp``,
    and ``runtime_environment`` copies the host environment into the child.
    So an operator's own shell can fail every sharded cell of a matrix for
    a reason no log explains.

    **The refusal reads the built argv, not the spec.** That is what keeps
    it from drifting away from ``megatron_stock/flags.py``: the day a flag
    list stops sending the flag, the refusal stops firing on its own.
    """

    def _resolve(self, environment: dict, **kwargs):
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", dict(_METADATA)),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            return _resolve_run(
                RunRequest(
                    gpu="0,1",
                    scenario_name="piper_megatron_stock",
                    ac_mode="none",
                    **kwargs,
                ),
                environment,
            )

    def _environment(self, value: str | None) -> dict:
        environment = {"PATH": os.environ["PATH"]}
        if value is not None:
            environment["CUDA_DEVICE_MAX_CONNECTIONS"] = value
        return environment

    def test_the_sharded_stock_argv_really_carries_the_flag(self) -> None:
        """The premise of every case below."""
        resolved = self._resolve(
            self._environment(None),
            parallelism=ParallelismSpec(dp=2, dense_sharding="zero3"),
        )
        self.assertIn("--use-megatron-fsdp", resolved[6]["baseline"])

    def test_the_forbidden_value_refuses_a_sharded_run(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "CUDA_DEVICE_MAX_CONNECTIONS"
        ) as raised:
            self._resolve(
                self._environment("1"),
                parallelism=ParallelismSpec(dp=2, dense_sharding="zero3"),
            )
        # The refusal names its own repair, as every other refusal here does.
        self.assertIn("Unset CUDA_DEVICE_MAX_CONNECTIONS", str(raised.exception))

    def test_another_value_and_an_unset_variable_both_pass(self) -> None:
        """Only the literal '1' is refused. Megatron asserts on that value
        alone, so refusing more would refuse a run Megatron accepts."""
        for value in ("8", None):
            with self.subTest(value=value):
                self._resolve(
                    self._environment(value),
                    parallelism=ParallelismSpec(dp=2, dense_sharding="zero3"),
                )

    def test_a_zero1_run_is_untouched_by_the_variable(self) -> None:
        """``zero1`` shards through the optimizer, not through Megatron-FSDP.

        The argv carries ``--use-distributed-optimizer`` and no
        ``--use-megatron-fsdp``, so Megatron runs no assert on the variable
        and the refusal goes inert on its own. The test reads both flags,
        because a refusal that fired here would refuse a legal cell.
        """
        resolved = self._resolve(
            self._environment("1"),
            parallelism=ParallelismSpec(dp=2, dense_sharding="zero1"),
        )
        argv = resolved[6]["baseline"]
        self.assertIn("--use-distributed-optimizer", argv)
        self.assertNotIn("--use-megatron-fsdp", argv)

    def test_a_replicated_run_is_untouched_by_the_variable(self) -> None:
        """A replicated stock run sends no ``--use-megatron-fsdp``, so the
        assert never runs and the variable is none of this repo's business.
        Refusing it here would refuse a cell this suite has already run."""
        resolved = self._resolve(
            self._environment("1"), parallelism=ParallelismSpec(dp=2)
        )
        self.assertNotIn("--use-megatron-fsdp", resolved[6]["baseline"])


class RunBannerTests(unittest.TestCase):
    """The run banner names every comparability boundary the manifest gates.

    A boundary the manifest records and the screen does not is one the
    operator cannot see while the run is starting. ``--resume`` refuses a
    changed ``parallelism`` record, and ``dense_sharding`` sits inside it,
    so the banner has to name the parity beside the three degrees.

    The run is made to fail at once: the banner prints before any arm
    starts, so a fixture that trains nothing still emits it. What is under
    test is the summary line, not the failure.
    """

    def _summaries(self, spec: ParallelismSpec, gpu: str) -> list[str]:
        events: list[str] = []

        def failing_process(command, **kwargs):
            kwargs["stdout"].write("nothing trained\n")
            return SimpleNamespace(returncode=1)

        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", dict(_METADATA)),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            request = RunRequest(
                gpu=gpu,
                scenario_name="piper1b_megatron",
                arm_names=("titan_stock",),
                out_dir=Path(temporary) / "run",
                ac_mode="none",
                parallelism=spec,
            )
            with self.assertRaises(RuntimeError):
                execute_run(
                    request,
                    event_handler=lambda event: events.append(
                        event.message if event.kind == "summary" else ""
                    ),
                    process_runner=failing_process,
                    environment={"PATH": os.environ["PATH"]},
                )
        return [message for message in events if message]

    def test_the_trivial_spec_banner_names_the_replicated_parity(self) -> None:
        lines = self._summaries(TRIVIAL_SPEC, "0")
        self.assertIn(
            "parallelism: dp 1 x pp 1 (ep 1, world size 1, "
            "dense sharding replicate)",
            lines,
        )

    def _warnings(self, spec: ParallelismSpec, gpu: str) -> list[str]:
        return [
            line
            for line in self._summaries(spec, gpu)
            if line.startswith("WARNING: ")
        ]

    def test_a_zero1_run_at_dp_one_carries_both_warnings(self) -> None:
        """Legal, and a reader must not take the value at face value. The
        shard degree is 1 there, and one microbatch puts the gradient
        reduce-scatter inside the only backward pass."""
        warnings = self._warnings(ParallelismSpec(dense_sharding="zero1"), "0")
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("at dp 1" in line for line in warnings))
        self.assertTrue(any("ZeRO-2" in line for line in warnings))

    def test_the_warnings_land_before_the_banner(self) -> None:
        """``_resolve_run`` emits them before it probes the host, so they
        reach the operator before the run claims a GPU. The banner is
        filled in by that probe, so it is the marker to sort against."""
        lines = self._summaries(ParallelismSpec(dense_sharding="zero1"), "0")
        first_warning = min(
            index
            for index, line in enumerate(lines)
            if line.startswith("WARNING: ")
        )
        banner = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("GPU (PCI index):")
        )
        self.assertLess(first_warning, banner)

    def test_the_intended_mesh_carries_no_warning(self) -> None:
        """A warning that fired on the configuration this axis exists to run
        would teach an operator to ignore warnings."""
        self.assertEqual(
            self._warnings(ParallelismSpec(dp=2, dense_sharding="zero3"), "0,1"),
            [],
        )

    def test_the_sharded_parity_reaches_the_banner(self) -> None:
        """The line has to MOVE with the value.

        A banner that named the parity but always printed ``replicate``
        would pass the test above and tell the operator nothing.
        """
        lines = self._summaries(
            ParallelismSpec(dp=2, dense_sharding="zero3"), "0,1"
        )
        self.assertIn(
            "parallelism: dp 2 x pp 1 (ep 1, world size 2, "
            "dense sharding zero3)",
            lines,
        )


if __name__ == "__main__":
    unittest.main()
