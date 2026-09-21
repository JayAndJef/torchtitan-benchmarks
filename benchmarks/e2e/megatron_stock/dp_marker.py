"""The data-parallel line, printed from the wrapper Megatron really built.

``install_data_parallel_marker`` wraps ``setup_model_and_optimizer`` and
prints one line per rank naming the wrapper, the sharding strategy and the
optimizer class the engine chose. A line derived from the harness request
instead would pass arm rule 12 on a run that silently lost its wrapper.

The shim restores Megatron after the first call, and it does nothing at one
data-parallel rank. Every torch and megatron import sits inside a function.
"""

from __future__ import annotations

from typing import Any

from benchmarks.e2e.megatron_stock.flags import NO_SHARD_STRATEGY
from benchmarks.e2e.megatron_stock.markers import DATA_PARALLEL_LINE


CHAINED_OPTIMIZERS_ATTRIBUTE = "chained_optimizers"
"""Megatron's own attribute for the members of a chain.

``ChainedOptimizer.__init__`` sets it, so the name is read off Megatron
rather than guessed. A submodule bump that renames it makes a chained run
print the outer class alone, and arm rule 12 then refuses the run, because
the expected string names the members.
"""


def optimizer_class_name(optimizer: Any) -> str:
    """The optimizer name the data-parallel line states.

    A bare optimizer states its own class.

    **A chain states its members too.** The standard path ends with an
    unconditional ``ChainedOptimizer(optimizers)``, so ``zero 0`` and
    ``zero 1`` both reach this function with a chain. That path always
    holds the dense optimizer. It adds a second member where
    TransformerEngine marked a weight for the expert process groups, which
    this argv reaches above expert degree 1 alone. Every member takes
    ``DistributedOptimizer`` under
    ``use_distributed_optimizer`` and
    ``Float16OptimizerWithFloat16Params`` without it, because that flag is
    one value for the whole run. So a ZeRO-0 chain and a ZeRO-1 chain
    carry the same outer class, and the members are what separate them.

    The member names are deduplicated and sorted, so one string covers
    both cases: ``ChainedOptimizer[DistributedOptimizer]`` where every
    member agrees, and ``ChainedOptimizer[A+B]`` where they do not.

    **An empty chain raises.** ``ChainedOptimizer`` accepts an empty list,
    for a rank that holds no trainable parameter. No path in this arm
    builds one. Such a chain names no class, so the line would state a
    ZeRO level nothing observed.
    """
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
    """Print the data-parallel line from the wrapper Megatron really built.

    Below two data-parallel ranks there is nothing to observe and nothing
    is printed: arm rule 12's data-parallel marker applies above ``dp`` 1.
    Spec rule 9 makes ``ep`` divide ``dp``, so every expert split above 1
    also has ``dp`` above 1 and reaches this line.

    Above it, this wraps ``setup_model_and_optimizer`` and reads the model
    it returns. **It raises when no chunk carries a data-parallel
    wrapper.** Two ranks that never reduce their gradients train two models
    and report about twice the true throughput, and every other rule passes
    -- including arm rule 13, because stock Megatron all-reduces the loss
    over the same group every step.

    **The check accepts ``_BaseDataParallel``, and the line names the
    class.** Megatron picks the wrapper from the arguments
    (``training.py``): ``DistributedDataParallel``,
    ``FullyShardedDataParallel`` or the torch FSDP2 wrapper. Each of them
    derives directly from ``_BaseDataParallel``
    (``data_parallel_base.py``). None of them derives from another. So a
    check against ``DistributedDataParallel`` alone raises on an honest
    sharded run.
    **``FullyShardedDataParallel`` is a factory function and not a class**
    (``mcore_fsdp_adapter.py``, whose own docstring says so), so
    ``isinstance`` against it raises ``TypeError``; the version classes are
    what a type check may name.

    Accepting the base class alone would prove less than the narrow check
    did. Printing ``type(chunk).__name__`` restores that and adds what the
    ZeRO level needs: the class name says which mechanism ran.

    **The sharding strategy printed is the one the run acts on.**
    ``data_parallel_sharding_strategy`` reaches every ``ddp_config``,
    because Megatron's argparse defaults it, but Megatron reads it only
    under a sharded wrapper. This suite builds none, so the line states ``no_shard``. A
    line built from the raw field would say a replicated run sharded.

    **The expert degree is read from the group, and a disagreement
    raises.** ``initialize_model_parallel`` has run by the time this
    replacement fires, so ``mpu.get_expert_model_parallel_world_size()``
    reports the group that really exists. The argv states the degree twice
    -- once as ``--expert-model-parallel-size`` and once on the mesh line
    -- and neither statement proves it. This is the same cross-check
    ``data.py`` already applies to the data-parallel degree.

    ``overlap_grad_reduce`` and ``grad_reduce_in_fp32`` come from the
    wrapper's own ``ddp_config``, not from the arguments.

    **The optimizer class is read here too, and it raises when it is
    absent.** ``setup_model_and_optimizer`` returns the optimizer beside
    the model, and its class is the one observation that separates
    ``zero 1`` from ``zero 0``: both keep the ``DistributedDataParallel``
    wrapper, so the wrapper name proves nothing about the optimizer state.
    A run that lost ``--use-distributed-optimizer`` builds
    ``Float16OptimizerWithFloat16Params`` and fails arm rule 12 here,
    rather than publishing ZeRO-0 memory under a ZeRO-1 label.

    **A mixture of experts gets a ``ChainedOptimizer``, and the line names
    its members.** ``optimizer_class_name`` does that. The outer class is
    the same under every ZeRO level, so the chain's own name
    separates none of them.
    """
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
        # The optimizer class is the one observation that separates zero1
        # from replicate, because both keep the same wrapper.
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
