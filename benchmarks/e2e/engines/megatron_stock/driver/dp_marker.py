"""The shim that prints the data-parallel line from the wrapper and the optimizer that Megatron built.

This module imports torch and megatron inside its functions alone.
"""

from __future__ import annotations

from typing import Any

from benchmarks.e2e.engines.megatron_stock.flags import NO_SHARD_STRATEGY
from benchmarks.e2e.engines.megatron_stock.driver.markers import DATA_PARALLEL_LINE


CHAINED_OPTIMIZERS_ATTRIBUTE = "chained_optimizers"
"""The attribute of Megatron's ``ChainedOptimizer`` that holds its members."""


def optimizer_class_name(optimizer: Any) -> str:
    """The class name of ``optimizer``, and of the members of a chain in sorted order; an empty chain raises."""
    name = type(optimizer).__name__
    members = getattr(optimizer, CHAINED_OPTIMIZERS_ATTRIBUTE, None)
    if members is None:
        return name
    if not members:
        raise RuntimeError(
            f"megatron returned a {name} with no member optimizer, so no "
            "line can name the class that holds the optimizer state; the "
            "run would record a ZeRO level it did not have"
        )
    inner = "+".join(sorted({type(member).__name__ for member in members}))
    return f"{name}[{inner}]"


def install_data_parallel_marker(
    *, data_parallel_size: int, expert_parallel_size: int = 1
) -> None:
    """Wrap ``setup_model_and_optimizer`` once, so that it prints the data-parallel line above one data-parallel rank."""
    if data_parallel_size <= 1:
        return

    import megatron.core.parallel_state as mpu
    import megatron.training.training as megatron_training
    from megatron.core.distributed.data_parallel_base import (
        _BaseDataParallel,
    )

    original = megatron_training.setup_model_and_optimizer

    def replacement(*args: Any, **keywords: Any) -> Any:
        result = original(*args, **keywords)
        model = result[0]
        chunks = model if isinstance(model, list) else [model]
        wrapped = [
            chunk for chunk in chunks if isinstance(chunk, _BaseDataParallel)
        ]
        if not wrapped:
            raise RuntimeError(
                f"the data-parallel degree is {data_parallel_size} and no "
                "model chunk carries a data-parallel wrapper, so "
                "no gradient is reduced; the ranks would train separate "
                "models and report about "
                f"{data_parallel_size}x the true throughput"
            )
        optimizer = result[1] if len(result) > 1 else None
        if optimizer is None:
            raise RuntimeError(
                "megatron returned no optimizer from "
                "setup_model_and_optimizer, so no line can name the class "
                "that holds the optimizer state; the run would record a "
                "ZeRO level it did not have"
            )
        config = wrapped[0].ddp_config
        built_expert_size = mpu.get_expert_model_parallel_world_size()
        if built_expert_size != expert_parallel_size:
            raise RuntimeError(
                "megatron built an expert-model-parallel group of "
                f"{built_expert_size} rank(s) and the arguments say "
                f"{expert_parallel_size}; the log would name one expert "
                "split and the run would do another"
            )
        print(
            DATA_PARALLEL_LINE.format(
                wrapper=type(wrapped[0]).__name__,
                dp=data_parallel_size,
                overlap=config.overlap_grad_reduce,
                fp32=config.grad_reduce_in_fp32,
                sharding=NO_SHARD_STRATEGY,
                expert=built_expert_size,
                optimizer=optimizer_class_name(optimizer),
            ),
            flush=True,
        )
        megatron_training.setup_model_and_optimizer = original
        return result

    megatron_training.setup_model_and_optimizer = replacement
