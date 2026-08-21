"""The per-arm build probe: a requirement the workload decides.

``KernelArm.requires_gcc_toolset`` is a property of the **host**, and the
parent answers it before it knows the shape. It is the only requirement the
roster could state until now, and two things were impossible because of that.

* **A sequence sweep.** An unfused attention arm's dense score tensor grows
  with the square of the sequence length, so there is a sequence at which it
  cannot be built. Before this, the arm was built anyway, the builder raised,
  and the whole scenario died -- ``run_correctness_pass`` builds every arm in
  one process and catches nothing.
* **Any arm that skips for a runtime reason** rather than a compile-time one.

``KernelArm.requirement`` is a dotted path to a parent-side predicate called
with ``(shape, workload)``. It returns the reason the arm cannot run here, or
``None``. The predicates below live in this module, which is exactly what a
real one must do: parent-side and torch-free, because the parent must never
import an ``operations/`` module.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kernel.runner import resolve_arm_skips
from benchmarks.kernel.schema import (
    CorrectnessCheck,
    KernelArm,
    KernelScenario,
    KernelWorkload,
    resolve_shape_and_workload,
    resolve_symbol,
)
from benchmarks.models.piper_qwen3.shape import PiperShape


# Two predicates with the contract's two answers, and one that models the
# case that forced the field: a limit the workload crosses.
def always_runnable(shape: PiperShape, workload: KernelWorkload) -> None:
    return None


def refuses_above_seq_512(
    shape: PiperShape, workload: KernelWorkload
) -> str | None:
    if workload.seq_len > 512:
        return (
            f"a dense score tensor at seq {workload.seq_len} exceeds this "
            "arm's budget"
        )
    return None


def _arm(
    name: str,
    *,
    requirement: str | None = None,
    reference: str | None = None,
) -> KernelArm:
    return KernelArm(
        name=name,
        description="synthetic",
        builder=f"tests.does_not_exist:{name}",
        modes=("forward",),
        requirement=requirement,
        compiled=True,
        correctness=(
            ()
            if reference is None
            else (
                CorrectnessCheck(
                    kind="tolerance",
                    reference=reference,
                    outputs=("out",),
                    max_rel_l2=1e-2,
                ),
            )
        ),
    )


HERE = "tests.test_kernel_arm_requirements"


def _scenario() -> KernelScenario:
    return KernelScenario(
        name="attention_core",
        description="synthetic",
        inputs_builder="tests.does_not_exist:inputs",
        reference_builder=None,
        baseline_arm="anchor",
        arms=(
            _arm("anchor", requirement=f"{HERE}:always_runnable"),
            _arm("dense", requirement=f"{HERE}:refuses_above_seq_512"),
            _arm("gated_on_dense", reference="dense"),
            _arm("independent", reference="anchor"),
        ),
    )


def _skips(seq_len: int) -> dict[str, str]:
    shape, workload = resolve_shape_and_workload(seq_len=seq_len)
    return resolve_arm_skips(
        _scenario(),
        compiler_unavailable=None,
        shape=shape,
        workload=workload,
    )


class ArmRequirementTests(unittest.TestCase):
    def test_no_arm_is_skipped_where_every_requirement_is_met(self) -> None:
        self.assertEqual(_skips(512), {})

    def test_an_arm_the_workload_rules_out_is_skipped_by_name(self) -> None:
        """And it is never built, so no builder has to raise.

        That is what makes this a probe rather than a rescue. Catching the
        builder's exception instead would turn a bug into a skipped arm: the
        roster would shorten for a reason nobody declared and the run would
        still exit zero.
        """
        skipped = _skips(1024)
        self.assertIn("dense", skipped)
        self.assertIn("exceeds this arm's budget", skipped["dense"])
        self.assertNotIn("anchor", skipped)
        self.assertNotIn("independent", skipped)

    def test_the_reason_is_the_predicate_s_own_words(self) -> None:
        """It reaches results.json as ``status_reason``.

        A reader of the file holds no registry, so a skip that did not say
        why would be indistinguishable from an arm nobody declared.
        """
        self.assertEqual(
            _skips(2048)["dense"],
            "a dense score tensor at seq 2048 exceeds this arm's budget",
        )

    def test_a_workload_skip_closes_over_correctness_references(self) -> None:
        """An arm whose reference is skipped is skipped too.

        The closure existed for the compiler skip and had never fired: no
        arm gates against ``titan/te``. A workload skip reaches it, because
        the arm it drops is a reference.
        """
        skipped = _skips(1024)
        self.assertIn("gated_on_dense", skipped)
        self.assertIn("'dense' is skipped", skipped["gated_on_dense"])
        self.assertNotIn("independent", skipped)

    def test_the_predicate_is_reached_by_a_dotted_path(self) -> None:
        """Resolution by string is what keeps the parent torch-free.

        A requirement that lived beside its family's builders would put an
        ``operations/`` module -- and torch behind it -- into the parent's
        import graph, which per-arm process isolation cannot have.
        """
        predicate = resolve_symbol(f"{HERE}:refuses_above_seq_512")
        shape, workload = resolve_shape_and_workload(seq_len=1024)
        self.assertIsNone(always_runnable(shape, workload))
        self.assertIsInstance(predicate(shape, workload), str)

    def test_a_declared_arm_carries_no_requirement_by_default(self) -> None:
        """Most arms have none, and an absent one is never a skip."""
        self.assertIsNone(_arm("plain").requirement)
        self.assertEqual(_skips(512), {})


if __name__ == "__main__":
    unittest.main()
