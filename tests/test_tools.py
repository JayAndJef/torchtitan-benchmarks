"""Structural tests for the developer scripts outside the Python packages.

Three things are pinned here:

* the pre-push hook, which runs the CPU suite before a push leaves the
  machine, and the ``sync.sh`` step that installs it;
* the guards of the Slurm matrix driver that need no Slurm and no GPU; and
* the module list of the continuous-integration workflow, which must name
  test modules that exist and that a host without the machine-learning
  stack can import.

It also runs the cell slug of the matrix script, whose failure is a silent
refusal of every pass.
"""

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_import_boundaries import REPO_ROOT

PRE_PUSH = REPO_ROOT / "tools" / "pre-push.sh"
SYNC = REPO_ROOT / "sync.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"
RUN_MATRIX = REPO_ROOT / "tools" / "run_matrix.sh"
RUN_MATRIX_CELL = REPO_ROOT / "tools" / "run_matrix_cell.sh"
MATRIX_JOB = REPO_ROOT / "tools" / "matrix_job.sbatch"

FORBIDDEN_IMPORTS = ("torch", "transformer_engine")
"""Module roots that a hosted runner cannot install."""

_MODULE_REFERENCE = re.compile(r"\btests\.(test_\w+)\b")


def workflow_modules() -> tuple[str, ...]:
    """The test modules the workflow runs, in the order it names them."""
    return tuple(_MODULE_REFERENCE.findall(WORKFLOW.read_text()))


def _module_path(name: str) -> Path | None:
    """The file of ``name`` if it is an in-repo module, else ``None``."""
    base = REPO_ROOT.joinpath(*name.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def import_closure(start: str) -> frozenset[str]:
    """Every module name reachable from ``start`` through ``ast`` imports.

    The walk follows ``import`` and ``from`` statements anywhere in a file,
    not at module scope alone, and it stops at the first module that is not
    in this repository. The result therefore holds the in-repo modules plus
    the outside names any of them reaches.
    """
    reached: set[str] = set()
    pending = [start]
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        path = _module_path(name)
        if path is None:
            continue
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parts = package.split(".")
                    root = ".".join(parts[: len(parts) - node.level + 1])
                    target = f"{root}.{node.module}" if node.module else root
                else:
                    target = node.module or ""
                pending.append(target)
                pending.extend(f"{target}.{alias.name}" for alias in node.names)
    return frozenset(reached)


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


class MatrixScriptTests(unittest.TestCase):
    """The refusals of the Slurm matrix driver that need no Slurm and no GPU."""

    def _driver(self, **overrides: str) -> subprocess.CompletedProcess:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("SLURM_")
        }
        env.update(DRY_RUN="1", NGPUS="1", TIME="10")
        env.update(overrides)
        return subprocess.run(
            ["bash", str(RUN_MATRIX)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_the_scripts_are_executable_bash(self) -> None:
        for script in (RUN_MATRIX, RUN_MATRIX_CELL, MATRIX_JOB):
            with self.subTest(script=script.name):
                self.assertTrue(os.access(script, os.X_OK))
                first = script.read_text().splitlines()[0]
                self.assertEqual(first, "#!/usr/bin/env bash")
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_the_job_template_refuses_its_placeholder_root(self) -> None:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("SLURM_")
        }
        env["SLURM_SUBMIT_DIR"] = str(REPO_ROOT)
        result = subprocess.run(
            ["bash", str(MATRIX_JOB)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("set ROOT", result.stderr)

    def test_the_driver_refuses_the_reserved_partitions(self) -> None:
        for partition in ("placeholder", "exceptions", "admin"):
            with self.subTest(partition=partition):
                result = self._driver(PARTITION=partition)
                self.assertEqual(result.returncode, 1)
                self.assertIn("run-gpu-job", result.stderr)

    def test_the_driver_refuses_the_physical_gpu_variable(self) -> None:
        result = self._driver(GPU="4")
        self.assertEqual(result.returncode, 1)
        self.assertIn("NGPUS", result.stderr)

    def test_the_driver_needs_a_time_limit(self) -> None:
        result = self._driver(TIME="")
        self.assertEqual(result.returncode, 1)
        self.assertIn("TIME is required", result.stderr)


STUB_RUN_BENCH = """#!/usr/bin/env bash
out=""
previous=""
for word in "$@"; do
    [ "$previous" = --out ] && out="$word"
    previous="$word"
done
echo "stub run_bench.sh $*"
mkdir -p "$out"
case "$STUB_MODE" in
    ok) echo '{"warnings": ["stub warning"]}' >"$out/results.json" ;;
    fail) exit 1 ;;
    contaminate)
        echo '{"warnings": []}' >"$out/results.json"
        echo "FOREIGN_PID stub" >>"$out.watch" ;;
esac
"""
"""A ``run_bench.sh`` that writes the outcome ``STUB_MODE`` names, and no more."""

FAKE_NVIDIA_SMI = """#!/usr/bin/env bash
for arg in "$@"; do
    case "$arg" in
        --query-gpu=index) echo 0 ;;
        --query-gpu=pci.bus_id) echo "$FAKE_BUS_ID" ;;
        --query-gpu=memory.used) echo 0 ;;
        --query-gpu=*) echo "0, Fake GPU, GPU-fake, $FAKE_BUS_ID, 0.0" ;;
    esac
done
"""
"""An ``nvidia-smi`` that shows one idle card at ``FAKE_BUS_ID``."""

ABSENT_BUS_ID = "0000FFFF:00:00.0"
"""A bus id in a PCI domain that no host has, so its NUMA node reads as unknown."""


class MatrixCellTests(unittest.TestCase):
    """The outcomes of the cell runner, against a stub run and a fake card.

    Each test copies the cell runner into a temporary repository, beside a
    stub ``run_bench.sh``, and commits both. A fake ``nvidia-smi`` and a fake
    load average stand in for the job, so no test needs Slurm or a GPU.
    """

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.repo = base / "repo"
        (self.repo / "tools").mkdir(parents=True)
        self.script = self.repo / "tools" / "run_matrix_cell.sh"
        self.script.write_text(RUN_MATRIX_CELL.read_text())
        self.script.chmod(0o755)
        stub = self.repo / "run_bench.sh"
        stub.write_text(STUB_RUN_BENCH)
        stub.chmod(0o755)
        (self.repo / ".gitignore").write_text(".venv\n")
        (self.repo / ".venv").symlink_to(REPO_ROOT / ".venv")
        bin_dir = base / "bin"
        bin_dir.mkdir()
        smi = bin_dir / "nvidia-smi"
        smi.write_text(FAKE_NVIDIA_SMI)
        smi.chmod(0o755)
        self.loadavg = base / "loadavg"
        self.loadavg.write_text("1.00 1.00 1.00 1/100 1\n")
        self.root = base / "root"
        self.root.mkdir()

        self.env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("SLURM_", "GIT_"))
        }
        self._git("init", "-q")
        self._git("add", "-A")
        self._git(
            "-c", "user.name=test", "-c", "user.email=test@example.com",
            "commit", "-q", "-m", "stub",
        )
        self.env.update(
            PATH=f"{bin_dir}{os.pathsep}{self.env['PATH']}",
            SLURM_JOB_ID="1",
            MATRIX_REV=self._git("rev-parse", "HEAD").strip(),
            MATRIX_LOADAVG=str(self.loadavg),
            FAKE_BUS_ID=ABSENT_BUS_ID,
            STUB_MODE="ok",
            IDLE_SETTLE="1",
            IDLE_POLL="1",
            IDLE_MAX_WAIT="1",
            WATCH_INTERVAL="1",
        )

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.repo), *args],
            env=self.env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    def _cell(
        self, name: str = "cell", *flags: str, prefix: tuple[str, ...] = ()
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*prefix, "bash", str(self.script), str(self.root), name,
             "run", "0", "--scenario", "engines", *flags],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def _status(self, name: str = "cell") -> str:
        return (self.root / f"{name}.status").read_text().strip()

    def _moved_aside(self, kind: str) -> list[str]:
        """The moved-aside names of the cell, with the time stamp cut out."""
        return sorted(
            re.sub(r"-\d{8}T\d{6}Z", "", path.name)
            for path in self.root.glob(f"cell.{kind}-*")
        )

    def test_a_passing_run_is_ok_and_repeats_its_warnings(self) -> None:
        result = self._cell()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self._status(), "OK")
        self.assertTrue((self.root / "cell" / "results.json").is_file())
        log = (self.root / "sweep.log").read_text()
        self.assertIn("STATUS cell OK", log)
        self.assertIn("results.json warnings: stub warning", log)
        self.assertIn(
            f"--scenario engines --out {self.root}/cell",
            (self.root / "cell.log").read_text(),
        )

    def test_a_done_cell_is_skipped(self) -> None:
        (self.root / "cell").mkdir()
        (self.root / "cell" / "results.json").write_text("{}")
        self.env["STUB_MODE"] = "fail"
        result = self._cell()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self._status(), "OK(existing)")
        self.assertFalse((self.root / "cell.log").exists())

    def test_the_inputs_are_refused(self) -> None:
        cases = {
            "a bad name": (("bad/name",), "must match"),
            "--out": (("cell", "--out", "elsewhere"), "--out"),
            "--resume=": (("cell", "--resume=elsewhere"), "--resume"),
        }
        for case, ((name, *flags), message) in cases.items():
            with self.subTest(case=case):
                result = self._cell(name, *flags)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stdout + result.stderr)
                self.assertFalse((self.root / "cell.log").exists())

    def test_a_missing_rev_is_refused(self) -> None:
        del self.env["MATRIX_REV"]
        result = self._cell()
        self.assertEqual(result.returncode, 2)
        self.assertIn("MATRIX_REV is not set", result.stdout)
        self.assertEqual(self._status(), "ERROR")

    def test_the_cell_runner_refuses_to_run_outside_a_job(self) -> None:
        del self.env["SLURM_JOB_ID"]
        result = self._cell()
        self.assertEqual(result.returncode, 2)
        self.assertIn("only inside a Slurm job", result.stdout)
        self.assertEqual(self._status(), "ERROR")

    def test_a_failed_run_moves_aside(self) -> None:
        self.env["STUB_MODE"] = "fail"
        result = self._cell()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._status(), "FAIL(rc=1)")
        self.assertFalse((self.root / "cell").exists())
        self.assertEqual(
            self._moved_aside("failed"),
            [
                "cell.failed",
                "cell.failed.log",
                "cell.failed.rc",
                "cell.failed.watch",
            ],
        )

    def test_a_flagged_run_is_contaminated_and_moves_aside(self) -> None:
        self.env["STUB_MODE"] = "contaminate"
        result = self._cell()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._status(), "CONTAMINATED")
        self.assertFalse((self.root / "cell").exists())
        self.assertEqual(
            self._moved_aside("contaminated"),
            [
                "cell.contaminated",
                "cell.contaminated.log",
                "cell.contaminated.rc",
                "cell.contaminated.watch",
            ],
        )

    def test_a_busy_host_runs_the_cell_and_flags_it(self) -> None:
        """The load stays between IDLE_LOAD and CONTENDED_LOAD, so the gate
        times out and the watchdog does not condemn the cell."""
        self.loadavg.write_text("100.00 100.00 100.00 1/100 1\n")
        result = self._cell()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self._status(), "OK(load-flagged)")
        self.assertIn(
            "LOAD-GATE-TIMEOUT load1=100", (self.root / "sweep.log").read_text()
        )

    def test_a_card_on_a_node_without_job_cpus_is_refused(self) -> None:
        """The test pins the runner to node 0 and gives it a card on node 1."""
        node0 = Path("/sys/devices/system/node/node0/cpulist")
        bus = next(
            (
                path.parent.name
                for path in sorted(Path("/sys/bus/pci/devices").glob("*/numa_node"))
                if path.read_text().strip() == "1"
            ),
            None,
        )
        if shutil.which("taskset") is None or not node0.exists() or bus is None:
            self.skipTest("the host has no taskset, no node 0, or no PCI device on node 1")
        cpus = os.sched_getaffinity(0) & set(_expand_cpulist(node0.read_text()))
        if not cpus:
            self.skipTest("this process may use no CPU of node 0")
        self.env["FAKE_BUS_ID"] = "0000" + bus.upper()
        pin = ("taskset", "-c", ",".join(str(cpu) for cpu in sorted(cpus)))
        result = self._cell(prefix=pin)
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertEqual(self._status(), "PLACEMENT")
        self.assertFalse((self.root / "cell.log").exists())


def _expand_cpulist(text: str) -> list[int]:
    """The CPU numbers of a sysfs list such as ``0-3,8``."""
    cpus = []
    for part in text.strip().split(","):
        low, _, high = part.partition("-")
        cpus.extend(range(int(low), int(high or low) + 1))
    return cpus


class WorkflowModuleListTests(unittest.TestCase):
    def test_the_workflow_names_at_least_five_modules(self) -> None:
        self.assertGreaterEqual(len(workflow_modules()), 5)

    def test_every_named_module_exists(self) -> None:
        missing = [
            name
            for name in workflow_modules()
            if not (REPO_ROOT / "tests" / f"{name}.py").is_file()
        ]
        self.assertEqual(
            missing,
            [],
            "The workflow names test modules that do not exist: "
            + ", ".join(missing),
        )

    def test_no_named_module_reaches_the_machine_learning_stack(self) -> None:
        offenders = []
        for name in workflow_modules():
            closure = import_closure(f"tests.{name}")
            roots = {reached.partition(".")[0] for reached in closure}
            for forbidden in FORBIDDEN_IMPORTS:
                if forbidden in roots:
                    offenders.append(f"tests.{name} reaches {forbidden}")
        self.assertEqual(
            offenders,
            [],
            "A hosted runner installs neither of these. Remove the module "
            "from the workflow, or defer the import:\n" + "\n".join(offenders),
        )

    def test_the_workflow_does_not_fetch_the_submodules(self) -> None:
        self.assertIn("submodules: false", WORKFLOW.read_text())


def cell_slug(cell: str) -> str:
    """The directory name ``tools/run_matrix.sh`` gives ``cell``."""
    source = RUN_MATRIX.read_text()
    match = re.search(r"^cell_slug\(\) \{\n.*?^\}\n", source, re.M | re.S)
    assert match, "tools/run_matrix.sh has no cell_slug function"
    return subprocess.run(
        ["bash", "-c", match.group(0) + 'cell_slug "$1"', "bash", cell],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


class MatrixCellSlugTests(unittest.TestCase):
    PIPELINE_FLAGS_CELL = (
        "--scenario engines --model-size 30b-a3b-20l --dp 2 --pp 2 --ep 2 "
        "--zero 1 --pp-schedule 1F1B --pp-microbatch-size 4 --batch 8 "
        "--profile --arm megatron_stock "
        "--megatron-arg=--cross-entropy-loss-fusion "
        "--megatron-arg=--moe-permute-fusion "
        "--megatron-arg=--overlap-grad-reduce "
        "--megatron-arg=--overlap-param-gather"
    )

    def test_a_short_cell_keeps_its_whole_slug(self) -> None:
        self.assertEqual(
            cell_slug("--dp 2 --arm titan_eager"), "dp-2-arm-titan_eager"
        )

    def test_the_longest_derived_name_fits_name_max(self) -> None:
        """The 2026-09-30 pipeline cell slugged to 272 bytes, and every pass
        failed before it wrote a log."""
        slug = cell_slug(self.PIPELINE_FLAGS_CELL)
        self.assertEqual(len(slug), 200)
        self.assertLessEqual(
            len(slug + ".contaminated-20260930T064353Z.log"), 255
        )

    def test_two_long_cells_that_share_a_prefix_differ(self) -> None:
        other = self.PIPELINE_FLAGS_CELL.replace(
            "overlap-param-gather", "use-flash-attn"
        )
        self.assertNotEqual(
            cell_slug(self.PIPELINE_FLAGS_CELL), cell_slug(other)
        )



if __name__ == "__main__":
    unittest.main()
