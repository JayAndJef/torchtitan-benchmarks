"""The path that the parallelism options take from the CLI to the child environment and the manifest."""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from click.testing import CliRunner

from benchmarks.artifacts.manifests import (
    MANIFEST_SCHEMA_VERSION,
    load_manifest,
    load_run_record,
    manifest_data,
    resume_mismatches,
)
from benchmarks.cli.e2e import (
    _PARALLELISM_OPTIONS,
    _execution_options,
    run_command,
)
from benchmarks.cli.main import cli
from benchmarks.e2e.axes import RequestedAxes, RunRequest
from benchmarks.e2e.engines.megatron_stock.config import MegatronStockConfig
from benchmarks.e2e.engines.torchtitan import mesh
from benchmarks.e2e.parallelism import TRIVIAL_SPEC, ParallelismSpec, describe
from benchmarks.e2e.registry import DEFAULT_AC_MODE, ENGINES
from benchmarks.e2e.runner import _resolve_run, execute_run
from tests.engine_helpers import TEST_METADATA, configured, run_spec, write_run_manifest
from benchmarks.execution import affinity, provenance
from benchmarks.execution.affinity import CpuPinning, resolve_cpu_pinning
from benchmarks.execution.devices import parse_devices
from benchmarks.execution.environment import (
    device_environment,
    runtime_environment,
)
from benchmarks.execution.paths import RuntimePaths
from benchmarks.execution.provenance import hardware_metadata


_METADATA = TEST_METADATA


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
        "--zero",
    )

    def _parameters(self) -> dict:
        return {
            option: parameter
            for parameter in run_command.params
            for option in parameter.opts
        }

    def test_the_ac_option_takes_its_default_from_the_registry(self) -> None:
        parameter = self._parameters()["--ac"]
        self.assertFalse(parameter.required)
        self.assertIn(f"[default: {DEFAULT_AC_MODE}]", parameter.help)
        self.assertEqual(DEFAULT_AC_MODE, "none")

    def test_the_megatron_treatments_default_off(self) -> None:
        config = MegatronStockConfig()
        self.assertEqual(config.p2p_sync, "off")
        self.assertEqual(config.nan_guard, "off")
        self.assertEqual(config.precision, "stock")

    def test_the_megatron_flag_gates_read_the_literal_value(self) -> None:
        """The token follows the treatment, never the default of the day.

        A gate spelled ``!= DEFAULT`` would invert the moment a default
        moved, and the argv would carry the opposite treatment under the
        same label.
        """
        from benchmarks.e2e.engines.megatron_stock.flags import (
            NAN_GUARD_FLAGS,
            NO_CHECK_FOR_NAN_FLAG,
        )

        self.assertEqual(NAN_GUARD_FLAGS["on"], ())
        self.assertEqual(NAN_GUARD_FLAGS["off"], (NO_CHECK_FOR_NAN_FLAG,))

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
             "zero"),
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
        for option in ("--ac", "--model-size"):
            with self.subTest(option=option):
                self.assertIsNotNone(parameters[option].envvar)

    def test_the_option_block_records_why(self) -> None:
        self.assertIn("no environment variable", _execution_options.__doc__)


class RequestTests(unittest.TestCase):
    """What the CLI hands ``RunRequest``."""

    def setUp(self) -> None:
        # These tests patch the runner, so they also patch its checks.
        check = mock.patch("benchmarks.cli.e2e.check_request")
        self.addCleanup(check.stop)
        check.start()

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
        request = self._request("0", "--scenario", "engines")
        self.assertIsNone(request.axes.parallelism)

    def test_the_gpu_string_is_kept_exactly_as_typed(self) -> None:
        # Roughly one hundred manifests record it as requested_gpu, and
        # CUDA_VISIBLE_DEVICES is set from the same value.
        for value in ("0", "0,1", "3,2"):
            with self.subTest(value=value):
                request = self._request(value, "--scenario", "engines")
                self.assertEqual(request.gpu, value)
                self.assertIsInstance(request.gpu, str)

    def test_the_zero_option_reaches_the_spec(self) -> None:
        request = self._request(
            "0,1",
            "--scenario",
            "engines",
            "--dp",
            "2",
            "--zero",
            1,
        )
        self.assertEqual(
            request.axes.parallelism,
            ParallelismSpec(dp=2, zero=1),
        )

    def test_the_zero_option_refuses_an_undeclared_value(
        self,
    ) -> None:
        """**Click refuses it, and the exit code is what says so.**

        A nonzero exit proves nothing here: a legal ``--zero 1`` also
        exits nonzero, because the run then starts and fails on this host
        for its own reasons. Click's usage error is exit 2, and it names
        the roster. Without the ``click.Choice`` the value would reach
        ``ParallelismSpec.__post_init__``, raise, and exit 1 -- a refusal
        in the right direction under the wrong code, which this assertion
        separates.

        **The value under test is 3, which is the RETIRED level.** Recorded
        cells carry it, so an operator who reads an old manifest can type
        it. It must reach the roster message.
        """
        result = CliRunner().invoke(
            cli,
            [
                "run",
                "0,1",
                "--scenario",
                "engines",
                "--dp",
                "2",
                "--zero",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("'0', '1'", result.output)

    def test_the_pipeline_options_build_one_spec(self) -> None:
        request = self._request(
            "0,1",
            "--scenario",
            "engines",
            "--pp",
            "2",
            "--pp-schedule",
            "1F1B",
            "--pp-microbatch-size",
            "1",
        )
        self.assertEqual(
            request.axes.parallelism,
            ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=1),
        )

    def test_an_option_left_out_takes_the_spec_default(self) -> None:
        request = self._request("0,1", "--scenario", "engines", "--dp", "2")
        self.assertEqual(request.axes.parallelism, ParallelismSpec(dp=2))

    def test_a_degree_below_one_is_refused_by_the_option(self) -> None:
        result = CliRunner().invoke(
            cli, ["run", "0", "--scenario", "engines", "--pp", "0"]
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
                ["run", "0,1", "--ac", "none", "--dp", "2"],
            )
        self.assertEqual(result.exit_code, 0, result.output)
        # Every swept scenario carries the same spec.
        self.assertTrue(seen)
        for request in seen:
            self.assertEqual(request.axes.parallelism, ParallelismSpec(dp=2))
            self.assertEqual(request.gpu, "0,1")


class DeviceEnvironmentTests(unittest.TestCase):
    def test_one_device(self) -> None:
        self.assertEqual(
            device_environment("0", world_size=1),
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "0",
                "NGPU": "1",
            },
        )

    def test_ngpu_follows_the_world_size(self) -> None:
        result = device_environment("0,1", world_size=2)
        self.assertEqual(result["NGPU"], "2")
        self.assertEqual(result["CUDA_VISIBLE_DEVICES"], "0,1")

    def test_a_world_size_below_one_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "world size"):
            device_environment("0", world_size=0)

    def test_the_runtime_environment_sets_no_device_key(self) -> None:
        paths = RuntimePaths.resolve(environment={"PATH": os.environ["PATH"]})
        result = runtime_environment(paths, environment={"PATH": os.environ["PATH"]})
        for key in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "NGPU", "LOG_RANK"):
            with self.subTest(key=key):
                self.assertNotIn(key, result)


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


def _manifest(
    parallelism: ParallelismSpec = TRIVIAL_SPEC, **megatron_fields: str
) -> dict:
    """The schema 20 manifest of a titan_eager and megatron_stock run at ``parallelism``."""
    arms = (
        ENGINES.arm("titan_eager"),
        configured(ENGINES.arm("megatron_stock"), **megatron_fields),
    )
    with tempfile.TemporaryDirectory() as temporary:
        out_dir = Path(temporary)
        write_run_manifest(
            out_dir, run_spec(profile=False, parallelism=parallelism), arms
        )
        return load_manifest(out_dir)


def _arm(manifest: dict, name: str) -> dict:
    """The record of the arm ``name`` in ``manifest``."""
    (record,) = [arm for arm in manifest["arms"] if arm["name"] == name]
    return record


class ManifestSchemaTwentyTests(unittest.TestCase):
    def test_the_schema_is_twenty(self) -> None:
        self.assertEqual(MANIFEST_SCHEMA_VERSION, 20)
        self.assertEqual(_manifest()["schema_version"], 20)

    def test_a_foreign_schema_is_refused_and_the_versions_are_named(
        self,
    ) -> None:
        for recorded in (8, 17, 21, None):
            with self.subTest(schema_version=recorded):
                with tempfile.TemporaryDirectory() as temporary:
                    out_dir = Path(temporary)
                    manifest = _manifest()
                    if recorded is None:
                        del manifest["schema_version"]
                    else:
                        manifest["schema_version"] = recorded
                    (out_dir / "manifest.json").write_text(json.dumps(manifest))
                    for read in (load_manifest, load_run_record):
                        with self.assertRaises(ValueError) as caught:
                            read(out_dir)
                        message = str(caught.exception)
                        self.assertIn(repr(recorded), message)
                        self.assertIn(str(MANIFEST_SCHEMA_VERSION), message)

    def test_a_current_manifest_reads_back_its_run_and_arms(self) -> None:
        spec = ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=2)
        manifest = _manifest(spec, precision="lean")
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            (out_dir / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(load_manifest(out_dir), manifest)
            record = load_run_record(out_dir)
        self.assertEqual(record.run, run_spec(profile=False, parallelism=spec))
        self.assertEqual(
            [arm.arm.name for arm in record.arms], ["titan_eager", "megatron_stock"]
        )
        self.assertEqual(
            record.arm("megatron_stock").arm,
            configured(ENGINES.arm("megatron_stock"), precision="lean"),
        )

    def test_the_trivial_spec_round_trips_through_json(self) -> None:
        recorded = _manifest()["run"]["parallelism"]
        self.assertEqual(recorded, describe(TRIVIAL_SPEC, local_batch_size=4))
        self.assertEqual(recorded["world_size"], 1)
        self.assertIsNone(recorded["pp_schedule"])
        self.assertEqual(recorded["zero"], 0)

    def test_a_pipelined_spec_round_trips_through_json(self) -> None:
        spec = ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=2)
        recorded = _manifest(spec)["run"]["parallelism"]
        self.assertEqual(recorded, describe(spec, local_batch_size=4))
        self.assertEqual(recorded["world_size"], 2)
        self.assertEqual(recorded["n_microbatches"], 2)

    def test_a_sharded_spec_records_the_level_and_no_engine_mesh(self) -> None:
        manifest = _manifest(ParallelismSpec(dp=2, zero=1))
        self.assertEqual(manifest["run"]["parallelism"]["zero"], 1)
        self.assertNotIn("dp_shard", manifest["run"]["parallelism"])
        self.assertEqual(
            _arm(manifest, "titan_eager")["execution_model"],
            "2-gpu-plain-bf16-dp2-zero1",
        )

    def test_an_omitted_field_is_a_type_error(self) -> None:
        with self.assertRaises(TypeError):
            manifest_data(
                scenario=ENGINES,
                hardware="test-gpu",
                metadata=_METADATA,
                arms=(),
            )

    def test_the_megatron_treatments_sit_in_the_arm_config(self) -> None:
        manifest = _manifest(p2p_sync="on", nan_guard="on", precision="lean")
        config = _arm(manifest, "megatron_stock")["config"]
        self.assertEqual(
            (config["p2p_sync"], config["nan_guard"], config["precision"]),
            ("on", "on", "lean"),
        )
        for key in ("p2p_sync", "nan_guard", "precision"):
            with self.subTest(key=key):
                self.assertNotIn(key, manifest["run"])
                self.assertNotIn(key, manifest["run"]["parallelism"])
                self.assertNotIn(key, _arm(manifest, "titan_eager")["config"])

    def test_only_the_precision_moves_the_megatron_execution_model(self) -> None:
        stock = _arm(_manifest(), "megatron_stock")["execution_model"]
        self.assertEqual(
            _arm(_manifest(p2p_sync="on", nan_guard="on"), "megatron_stock")[
                "execution_model"
            ],
            stock,
        )
        self.assertNotEqual(
            _arm(_manifest(precision="lean"), "megatron_stock")["execution_model"],
            stock,
        )


class ExecutionModelFollowsTheMeshTests(unittest.TestCase):
    """The TorchTitan execution model follows the run's own mesh."""

    def test_the_trivial_spec_records_the_single_gpu_string(self) -> None:
        self.assertEqual(
            _arm(_manifest(), "titan_eager")["execution_model"],
            "single-gpu-plain-bf16-no-fsdp",
        )

    def test_a_pipelined_spec_records_its_own_mesh(self) -> None:
        spec = ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=2)
        self.assertEqual(
            _arm(_manifest(spec), "titan_eager")["execution_model"],
            "2-gpu-plain-bf16-no-fsdp-pp2-1F1B",
        )

    def test_the_field_is_what_the_engine_composes(self) -> None:
        for spec in (
            TRIVIAL_SPEC,
            ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=2),
        ):
            with self.subTest(spec=spec):
                self.assertEqual(
                    _arm(_manifest(spec), "titan_eager")["execution_model"],
                    mesh.execution_model(spec),
                )


class ExecutionModelIsNotResumeGatedTests(unittest.TestCase):
    """A resume gates the parallelism block and not the execution model that derives from it."""

    ARMS = (ENGINES.arm("titan_eager"), ENGINES.arm("megatron_stock"))

    def test_a_changed_execution_model_alone_resumes(self) -> None:
        manifest = _manifest()
        for record in manifest["arms"]:
            record["execution_model"] = "something-else-entirely"
        self.assertEqual(
            resume_mismatches(
                manifest, run=run_spec(profile=False), arms=self.ARMS
            ),
            [],
        )

    def test_the_spec_it_derives_from_is_gated(self) -> None:
        self.assertEqual(
            resume_mismatches(
                _manifest(),
                run=run_spec(
                    profile=False,
                    parallelism=ParallelismSpec(pp=2, pp_schedule="1F1B"),
                ),
                arms=self.ARMS,
            ),
            ["run.parallelism"],
        )


class ResumeParallelismTests(unittest.TestCase):
    ARMS = (ENGINES.arm("titan_eager"), ENGINES.arm("megatron_stock"))

    def _mismatches(self, manifest: dict, parallelism: ParallelismSpec) -> list[str]:
        return resume_mismatches(
            manifest,
            run=run_spec(profile=False, parallelism=parallelism),
            arms=self.ARMS,
        )

    def test_a_different_mesh_refuses_a_resume(self) -> None:
        spec = ParallelismSpec(pp=2, pp_schedule="1F1B")
        manifest = _manifest(spec)
        self.assertEqual(self._mismatches(manifest, spec), [])
        for requested in (
            TRIVIAL_SPEC,
            ParallelismSpec(pp=2, pp_schedule="Interleaved1F1B"),
            ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=2),
        ):
            with self.subTest(requested=requested):
                self.assertEqual(
                    self._mismatches(manifest, requested), ["run.parallelism"]
                )

    def test_the_zero_level_alone_refuses_a_resume(self) -> None:
        recorded = ParallelismSpec(dp=2)
        requested = ParallelismSpec(dp=2, zero=1)
        self.assertIn(
            "run.parallelism", self._mismatches(_manifest(recorded), requested)
        )
        self.assertIn(
            "run.parallelism", self._mismatches(_manifest(requested), recorded)
        )

    def test_a_block_without_the_zero_key_refuses_a_resume(self) -> None:
        manifest = _manifest()
        del manifest["run"]["parallelism"]["zero"]
        self.assertIn("run.parallelism", self._mismatches(manifest, TRIVIAL_SPEC))


class ResumeMegatronConfigTests(unittest.TestCase):
    """A resume gates each Megatron treatment as one field of the arm's config."""

    def test_each_treatment_alone_refuses_a_resume(self) -> None:
        for field, values in (
            ("p2p_sync", ("on", "off")),
            ("nan_guard", ("on", "off")),
            ("precision", ("stock", "lean")),
        ):
            for recorded in values:
                for requested in values:
                    with self.subTest(
                        field=field, recorded=recorded, requested=requested
                    ):
                        mismatches = resume_mismatches(
                            _manifest(**{field: recorded}),
                            run=run_spec(profile=False),
                            arms=(
                                ENGINES.arm("titan_eager"),
                                configured(
                                    ENGINES.arm("megatron_stock"),
                                    **{field: requested},
                                ),
                            ),
                        )
                        self.assertEqual(
                            mismatches,
                            []
                            if recorded == requested
                            else [f"megatron_stock.config.{field}"],
                        )


_AXIS_KEYWORDS = tuple(field.name for field in fields(RequestedAxes))


class ResolveRunTests(unittest.TestCase):
    """``_resolve_run`` resolves the mesh."""

    def _resolve(self, scenario_name: str = "engines", **kwargs):
        kwargs.setdefault("ac_mode", "none")
        # One flat mapping per case, split into the axes and the rest.
        axes = RequestedAxes(
            **{
                name: kwargs.pop(name)
                for name in list(kwargs)
                if name in _AXIS_KEYWORDS
            }
        )
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", dict(_METADATA)),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            return _resolve_run(
                RunRequest(scenario_name=scenario_name, axes=axes, **kwargs),
                {"PATH": os.environ["PATH"]},
            )

    def test_a_single_gpu_run_resolves_to_the_trivial_spec(self) -> None:
        resolved = self._resolve(gpu="0")
        self.assertEqual(resolved.run.parallelism, TRIVIAL_SPEC)

    def test_a_named_trivial_spec_resolves_the_same_way(self) -> None:
        self.assertEqual(self._resolve(gpu="0", parallelism=TRIVIAL_SPEC).run.parallelism, TRIVIAL_SPEC)

    def test_a_mesh_that_does_not_fill_the_device_list_is_refused(self) -> None:
        # Rule 1: not "at most". An under-filled request would leave a GPU
        # idle and publish the number under the whole device list.
        with self.assertRaisesRegex(ValueError, "does not match"):
            self._resolve(gpu="0,1")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self._resolve(gpu="0", parallelism=ParallelismSpec(pp=2, pp_schedule="1F1B"))

    def test_a_legal_mesh_resolves(self) -> None:
        spec = ParallelismSpec(pp=2, pp_schedule="1F1B")
        self.assertEqual(self._resolve(gpu="0,1", parallelism=spec).run.parallelism, spec)

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
                    RunRequest(gpu="0,1", scenario_name="engines"),
                    {"PATH": os.environ["PATH"]},
                )


class RunBannerTests(unittest.TestCase):
    """The run banner names every comparability boundary the manifest gates.

    A boundary the manifest records and the screen does not is one the
    operator cannot see while the run is starting. ``--resume`` refuses a
    changed ``parallelism`` record, and ``zero`` sits inside it,
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
                axes=RequestedAxes(
                    ac_mode="none",
                    parallelism=spec,
                ),
                gpu=gpu,
                scenario_name="engines",
                arm_names=("titan_compiled",),
                out_dir=Path(temporary) / "run",
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
            "parallelism: dp 1 x pp 1 (ep 1, world size 1, zero 0)",
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
        warnings = self._warnings(ParallelismSpec(zero=1), "0")
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("at dp 1" in line for line in warnings))
        self.assertTrue(any("ZeRO-2" in line for line in warnings))

    def test_the_warnings_land_before_the_banner(self) -> None:
        """``_resolve_run`` emits them before it probes the host, so they
        reach the operator before the run claims a GPU. The banner is
        filled in by that probe, so it is the marker to sort against."""
        lines = self._summaries(ParallelismSpec(zero=1), "0")
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
            self._warnings(
                ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B", zero=1),
                "0,1,2,3",
            ),
            [],
        )

    def test_the_sharded_parity_reaches_the_banner(self) -> None:
        """The line has to MOVE with the value.

        A banner that named the parity but always printed ``zero 0``
        would pass the test above and tell the operator nothing.
        """
        lines = self._summaries(
            ParallelismSpec(dp=2, zero=1), "0,1"
        )
        self.assertIn(
            "parallelism: dp 2 x pp 1 (ep 1, world size 2, zero 1)",
            lines,
        )


if __name__ == "__main__":
    unittest.main()
