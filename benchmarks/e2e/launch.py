"""Build the training launch of one end-to-end benchmark arm.

This module holds one builder per engine. Each builder reads the run and
the arm's config and returns a ``Launch``, and each engine in
``benchmarks.e2e.engines`` calls its own builder.

At the trivial parallelism spec no ``--parallelism.*`` token appears.
``tests/test_megatron_stock_launch.py`` asserts that over every TorchTitan
arm of every scenario.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.e2e.engines.api import Arm, Launch, RunSpec
from benchmarks.e2e.passthrough import refuse_megatron_passthrough


STOCK_MEGATRON_DRIVER_MODULE = "benchmarks.e2e.megatron_stock.train"
"""What ``python -m`` starts for the stock Megatron arm.

Named once, so a test and the command builder cannot drift apart.
"""

STOCK_MEGATRON_PP_SCHEDULE = "1F1B"
"""The one pipeline schedule the stock driver implements.

Megatron-LM implements more, but this driver builds no model-chunk list, so
it runs ``forward_backward_pipelining_without_interleaving`` alone.
"""


def megatron_stock_launch(run: RunSpec, arm: Arm, arm_dir: Path) -> Launch:
    """The launch of one stock Megatron-LM arm.

    ``benchmarks/e2e/megatron_stock/flags.py`` builds every flag, and this
    function adds the driver module and the passthrough tokens. The
    refusals repeat the run checks, so a caller that builds a launch
    without a run gets a message that names the arm.
    """
    config = arm.config
    spec = run.parallelism
    if run.ac_mode != "none":
        raise ValueError(
            f"{arm.name}: the stock megatron arm runs without recompute; "
            f"ac mode {run.ac_mode!r} has no Megatron parity (use --ac none)"
        )
    if run.seed is None:
        raise ValueError(
            f"{arm.name}: megatron arms require a seeded workload"
        )
    if spec.pp > 1 and spec.pp_schedule != STOCK_MEGATRON_PP_SCHEDULE:
        raise ValueError(
            f"{arm.name}: the stock megatron driver implements "
            f"{STOCK_MEGATRON_PP_SCHEDULE!r} alone, and this run asks for "
            f"{spec.pp_schedule!r}"
        )
    # Below the refusals, so a refused request fails with its own message.
    from benchmarks.e2e.megatron_stock.flags import stock_megatron_flags

    refuse_megatron_passthrough(arm.name, config.extra_flags, spec.zero)
    return Launch(
        target=(
            "-m",
            STOCK_MEGATRON_DRIVER_MODULE,
            *stock_megatron_flags(run, config, arm_dir=str(arm_dir)),
            # Last, because Megatron's parser is last-wins.
            *config.extra_flags,
        ),
        processes="per_rank",
        pin=True,
    )
