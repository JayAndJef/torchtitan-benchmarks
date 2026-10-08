"""Structural tests for the developer scripts outside the Python packages.

Three things are pinned here:

* the pre-push hook, which runs the CPU suite before a push leaves the
  machine, and the ``sync.sh`` step that installs it;
* the matrix job template and the cell runner, whose outcomes the tests
  read against a stub run, with no Slurm and no GPU; and
* the module list of the continuous-integration workflow, which must name
  test modules that exist and that a host without the machine-learning
  stack can import.

The matrix tests need the synced ``.venv``, because the cell runner reads
``results.json`` with its interpreter. So the workflow does not run this
module.
"""

import ast
import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.registry import SCENARIOS
from tests.test_import_boundaries import REPO_ROOT

PRE_PUSH = REPO_ROOT / "tools" / "pre-push.sh"
SYNC = REPO_ROOT / "sync.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "tests.yml"
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
    """The two matrix scripts, and the guard of the job template."""

    def test_the_scripts_are_executable_bash(self) -> None:
        for script in (RUN_MATRIX_CELL, MATRIX_JOB):
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

    def test_every_template_cell_names_one_known_scenario_and_known_arms(self) -> None:
        """A cell that names an arm the branch lacks fails only in the job."""
        text = MATRIX_JOB.read_text()
        variables = dict(re.findall(r'^([A-Z]+)="([^"]*)"$', text, re.M))
        cells = re.findall(r"^cell (.+)$", text, re.M)
        self.assertGreater(len(cells), 0)
        names = []
        for cell in cells:
            words = re.sub(
                r"\$([A-Z]+)", lambda match: variables[match.group(1)], cell
            ).split()
            name, flags = words[0], words[1:]
            names.append(name)
            with self.subTest(cell=name):
                self.assertRegex(name, r"\A[A-Za-z0-9_-]+\Z")
                self.assertFalse(
                    [flag for flag in flags if flag.startswith(("--scenario=", "--arm="))]
                )
                self.assertEqual(flags.count("--scenario"), 1)
                scenario = flags[flags.index("--scenario") + 1]
                self.assertIn(scenario, sorted(SCENARIOS))
                arms = sorted(arm.name for arm in SCENARIOS[scenario].arms)
                for index, flag in enumerate(flags):
                    if flag == "--arm":
                        self.assertIn(flags[index + 1], arms)
        self.assertEqual(len(names), len(set(names)), "two cells share a name")


STUB_RUN_BENCH = """#!/usr/bin/env bash
out=""
previous=""
for word in "$@"; do
    [ "$previous" = --out ] && out="$word"
    previous="$word"
done
echo "stub run_bench.sh $*"
mkdir -p "$out"
sleep "${STUB_SLEEP:-0}"
case "$STUB_MODE" in
    ok) echo '{"warnings": ["stub warning"]}' >"$out/results.json" ;;
    fail) exit 1 ;;
    noresults) exit 0 ;;
    badwarnings) echo '{"warnings": [{"not": "text"}]}' >"$out/results.json" ;;
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
        --query-gpu=memory.used)
            [ -z "${FAKE_FAIL_MEM:-}" ] || { echo "fake memory failure" >&2; exit 9; }
            echo 0 ;;
        --query-compute-apps=*)
            [ -z "${FAKE_FAIL_APPS:-}" ] || { echo "fake apps failure" >&2; exit 9; } ;;
        --query-gpu=*) echo "0, Fake GPU, GPU-fake, $FAKE_BUS_ID, 0.0" ;;
    esac
done
"""
"""An ``nvidia-smi`` that shows one idle card at ``FAKE_BUS_ID``.

``FAKE_FAIL_MEM`` and ``FAKE_FAIL_APPS`` make the two watchdog queries fail,
and the checks before the run still pass.
"""

ABSENT_BUS_ID = "0000FFFF:00:00.0"
"""A bus id in a PCI domain that no host has, so its NUMA node reads as unknown."""

STAMP = re.compile(r"-\d{8}T\d{6}Z-j\d+")
"""The stamp of a renamed cell: the UTC time and the job id."""


def _expand_cpulist(text: str) -> set[int]:
    """The CPU numbers of a sysfs list such as ``0-3,8``."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        if part:
            low, _, high = part.partition("-")
            cpus.update(range(int(low), int(high or low) + 1))
    return cpus


def _node_cpus() -> dict[int, set[int]]:
    """The CPUs of each NUMA node of this host, read from sysfs."""
    return {
        int(path.parent.name[len("node"):]): _expand_cpulist(path.read_text())
        for path in Path("/sys/devices/system/node").glob("node[0-9]*/cpulist")
    }


def _bus_id_on(nodes: set[int]) -> str | None:
    """A PCI device on one of ``nodes``, as nvidia-smi spells a bus id."""
    for path in sorted(Path("/sys/bus/pci/devices").glob("*/numa_node")):
        if path.read_text().strip() in {str(node) for node in nodes}:
            return "0000" + path.parent.name.upper()
    return None


class MatrixCellTests(unittest.TestCase):
    """The outcomes of the cell runner, against a stub run and a fake card.

    Each test copies the cell runner into a temporary repository, beside a
    stub ``run_bench.sh``, and commits both. A fake ``nvidia-smi`` and a fake
    load average stand in for the job, so no test needs Slurm or a GPU. The
    fake card sits at a real PCI device on a NUMA node whose CPUs this
    process holds, because the runner refuses a card on an unknown node.
    """

    def setUp(self) -> None:
        held = os.sched_getaffinity(0)
        bus = _bus_id_on(
            {node for node, cpus in _node_cpus().items() if cpus & held}
        )
        if bus is None:
            self.skipTest("the host has no PCI device on a NUMA node of our CPUs")
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
        self.env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
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
            FAKE_BUS_ID=bus,
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
        self,
        name: str = "cell",
        *flags: str,
        devices: str = "0",
        prefix: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*prefix, "bash", str(self.script), str(self.root), name,
             "run", devices, "--scenario", "engines", *flags],
            env=self.env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def _status(self, name: str = "cell") -> str:
        return (self.root / f"{name}.status").read_text().strip()

    def _sweep(self) -> str:
        return (self.root / "sweep.log").read_text()

    def _renamed(self, kind: str, name: str = "cell") -> list[str]:
        """The renamed files of the cell, with the stamp cut out."""
        return sorted(
            STAMP.sub("", path.name) for path in self.root.glob(f"{name}.{kind}-*")
        )

    def _write_cell(self, status: str | None, name: str = "cell") -> None:
        """A finished cell of an earlier attempt, with ``status`` if any."""
        (self.root / name).mkdir()
        (self.root / name / "results.json").write_text("{}")
        (self.root / f"{name}.log").write_text("earlier attempt\n")
        if status is not None:
            (self.root / f"{name}.status").write_text(status + "\n")

    def test_a_passing_run_is_ok_and_repeats_its_warnings(self) -> None:
        result = self._cell()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self._status(), "OK")
        self.assertTrue((self.root / "cell" / "results.json").is_file())
        self.assertIn("STATUS cell OK", self._sweep())
        self.assertIn("results.json warnings: stub warning", self._sweep())
        self.assertIn(
            f"--scenario engines --out {self.root}/cell",
            (self.root / "cell.log").read_text(),
        )

    def test_a_done_cell_is_skipped_and_keeps_its_state(self) -> None:
        self._write_cell("OK(load-flagged)")
        self.env["STUB_MODE"] = "fail"
        result = self._cell()
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self._status(), "OK(load-flagged)")
        self.assertIn("STATUS cell OK(existing:OK(load-flagged))", self._sweep())
        self.assertEqual((self.root / "cell.log").read_text(), "earlier attempt\n")

    def test_results_without_an_ok_status_run_again(self) -> None:
        for name, status in (("nostatus", None), ("flagged", "CONTAMINATED")):
            with self.subTest(status=status):
                self._write_cell(status, name)
                result = self._cell(name)
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertEqual(self._status(name), "OK")
                self.assertEqual(
                    self._renamed("failed", name),
                    [f"{name}.failed", f"{name}.failed.log"],
                )
                self.assertIn(
                    "stub run_bench.sh", (self.root / f"{name}.log").read_text()
                )

    def test_a_partial_cell_of_an_earlier_job_is_renamed(self) -> None:
        (self.root / "cell").mkdir()
        (self.root / "cell" / "partial").write_text("x")
        (self.root / "cell.log").write_text("earlier attempt\n")
        result = self._cell()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(self._status(), "OK")
        self.assertEqual(self._renamed("failed"), ["cell.failed", "cell.failed.log"])
        self.assertFalse((self.root / "cell" / "partial").exists())
        self.assertNotIn("earlier attempt", (self.root / "cell.log").read_text())

    def test_a_rename_never_lands_on_a_name_that_exists(self) -> None:
        """The test takes the stamps of the next ten seconds in advance."""
        (self.root / "cell").mkdir()
        now = datetime.datetime.now(datetime.timezone.utc)
        for offset in range(10):
            moment = now + datetime.timedelta(seconds=offset)
            (self.root / f"cell.failed-{moment:%Y%m%dT%H%M%SZ}-j1").mkdir()
        result = self._cell()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertEqual(self._status(), "ERROR")
        self.assertIn("keeps its name", result.stdout)
        self.assertIn("a resubmit clears it", result.stdout)
        self.assertTrue((self.root / "cell").is_dir())
        self.assertEqual(list(self.root.glob("cell.failed-*/*")), [])

    def test_the_inputs_are_refused(self) -> None:
        cases = {
            "a slash in the name": (("bad/name",), "must match"),
            "a dot in the name": (("cell.log",), "must match"),
            "--out": (("cell", "--out", "elsewhere"), "--out"),
            "--resume=": (("cell", "--resume=elsewhere"), "--resume"),
        }
        for case, ((name, *flags), message) in cases.items():
            with self.subTest(case=case):
                result = self._cell(name, *flags)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stdout + result.stderr)
                self.assertFalse((self.root / "cell.log").exists())

    def test_a_device_that_the_job_lacks_is_refused(self) -> None:
        for devices, message in (("1", "sees no card 1"), ("0,1", "names 2")):
            with self.subTest(devices=devices):
                result = self._cell(devices=devices)
                self.assertEqual(result.returncode, 2)
                self.assertIn(message, result.stdout)
                self.assertEqual(self._status(), "ERROR")

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

    def test_a_failed_run_is_renamed(self) -> None:
        self.env["STUB_MODE"] = "fail"
        result = self._cell()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._status(), "FAIL(rc=1)")
        self.assertFalse((self.root / "cell").exists())
        self.assertEqual(
            self._renamed("failed"),
            ["cell.failed", "cell.failed.log", "cell.failed.rc", "cell.failed.watch"],
        )

    def test_a_run_without_results_is_renamed(self) -> None:
        self.env["STUB_MODE"] = "noresults"
        result = self._cell()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._status(), "FAIL(no-results)")
        self.assertFalse((self.root / "cell").exists())
        self.assertIn("cell.failed", self._renamed("failed"))

    def test_unreadable_warnings_rename_the_cell(self) -> None:
        self.env["STUB_MODE"] = "badwarnings"
        result = self._cell()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(self._status(), "ERROR")
        self.assertFalse((self.root / "cell").exists())
        self.assertIn("cell.failed", self._renamed("failed"))

    def test_a_flagged_run_is_contaminated_and_renamed(self) -> None:
        self.env["STUB_MODE"] = "contaminate"
        result = self._cell()
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._status(), "CONTAMINATED")
        self.assertFalse((self.root / "cell").exists())
        self.assertEqual(
            self._renamed("contaminated"),
            [
                "cell.contaminated",
                "cell.contaminated.log",
                "cell.contaminated.rc",
                "cell.contaminated.watch",
            ],
        )

    def test_a_blind_watchdog_contaminates_the_cell(self) -> None:
        """The run sleeps, so the watchdog samples the failed query in it."""
        self.env["STUB_SLEEP"] = "2"
        for failure, reason in (
            ("FAKE_FAIL_MEM", "WATCH-BLIND memory.used is unreadable"),
            ("FAKE_FAIL_APPS", "WATCH-BLIND compute-apps gpu=0: fake apps failure"),
        ):
            with self.subTest(failure=failure):
                self.env[failure] = "1"
                name = failure.lower()
                result = self._cell(name)
                del self.env[failure]
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertEqual(self._status(name), "CONTAMINATED")
                (watch,) = self.root.glob(f"{name}.contaminated-*.watch")
                self.assertIn(reason, watch.read_text())

    def test_a_busy_host_runs_the_cell_and_flags_it(self) -> None:
        """The load stays between IDLE_LOAD and CONTENDED_LOAD, so the gate
        times out and the watchdog does not condemn the cell."""
        self.loadavg.write_text("100.00 100.00 100.00 1/100 1\n")
        result = self._cell()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self._status(), "OK(load-flagged)")
        self.assertIn("LOAD-GATE-TIMEOUT load1=100 max=100", self._sweep())

    def test_a_card_on_an_unknown_node_is_refused(self) -> None:
        self.env["FAKE_BUS_ID"] = ABSENT_BUS_ID
        result = self._cell()
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertEqual(self._status(), "PLACEMENT")
        self.assertFalse((self.root / "cell.log").exists())

    def test_a_card_on_a_node_without_job_cpus_is_refused(self) -> None:
        """The test pins the runner to the CPUs of one node and gives it a
        card on a node that holds none of them."""
        if shutil.which("taskset") is None:
            self.skipTest("the host has no taskset")
        nodes = _node_cpus()
        held = os.sched_getaffinity(0)
        for node, cpus in sorted(nodes.items()):
            pinned = cpus & held
            others = {
                other
                for other, other_cpus in nodes.items()
                if other != node and not other_cpus & pinned
            }
            bus = _bus_id_on(others) if pinned else None
            if bus is not None:
                break
        else:
            self.skipTest("the host has no PCI device on a second NUMA node")
        self.env["FAKE_BUS_ID"] = bus
        pin = ("taskset", "-c", ",".join(str(cpu) for cpu in sorted(pinned)))
        result = self._cell(prefix=pin)
        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
        self.assertEqual(self._status(), "PLACEMENT")
        self.assertFalse((self.root / "cell.log").exists())


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


if __name__ == "__main__":
    unittest.main()
