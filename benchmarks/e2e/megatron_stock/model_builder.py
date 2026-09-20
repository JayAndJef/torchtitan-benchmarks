"""A stock GPT builder that counts its own parameters and prints the total.

Arm rule 11 (``benchmarks/e2e/validation.py``) needs
``size: <N> total parameters`` on every rank, with the shape's exact count
and a thousands separator. Stock Megatron prints a per-stage count in its
own format and no whole-model total, so the line has to come from
first-party code.

**The printed total is backed by a real count on the rank that prints it.**
The builder counts the parameters the model on this rank actually holds,
asserts that count against ``PiperShape.stage_param_count`` for this
pipeline degree and this stage, and prints the total only after the
assertion passes. A rank that built the wrong slice therefore fails the run
instead of printing a total it did not verify.

**No collective runs here.** An all-reduce inside a builder deadlocks if any
rank ever builds a different number of model chunks. This builder declines
the collective route, and pays for it with a per-stage identity rather than
a measured sum. ``tests/test_model_shape.py`` already
pins the arithmetic that the stage counts sum to ``param_count``.

``GPTModelConfig.builder`` is a ``ClassVar[str]`` dotted path
(Megatron's own GPT model provider), and
``ModelConfig.get_builder_cls`` imports it. ``BenchGPTModelConfig`` changes
that one string and adds no field, which is the same extension point
``megatron.post_training``'s ``ModelOptModelConfig`` uses.

This module imports megatron at module scope. That is legal here and only
here: ``ModelConfig.get_builder_cls`` imports it inside the training
process, where ``bootstrap.prepare()`` has already put Megatron-LM on
``sys.path``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from megatron.core import mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training import get_args
from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig

from benchmarks.e2e.megatron_stock.train import (
    MODEL_SIZE_LINE,
    STAGE_SIZE_LINE,
)
from benchmarks.models.piper_qwen3.shape import shape_by_name


@dataclass(kw_only=True)
class BenchGPTModelConfig(GPTModelConfig):
    """``GPTModelConfig`` with one changed string: the builder path.

    It adds no field, so ``gpt_config_from_args`` builds it from the same
    argument derivation the stock path uses.
    """

    builder: ClassVar[str] = (
        "benchmarks.e2e.megatron_stock.model_builder.CountingGPTModelBuilder"
    )


class CountingGPTModelBuilder(GPTModelBuilder):
    """The stock builder, plus the two parameter lines and their check."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> GPTModel:
        """Build one stage, count it, check it, and print the two lines.

        Refuses a virtual pipeline degree. ``stage_param_count`` describes
        one chunk per rank, so a virtual pipeline would compare a chunk
        against a whole stage and fail on an honest run.
        """
        config = self._model_config
        if config.virtual_pipeline_model_parallel_size is not None:
            raise ValueError(
                "the stock megatron driver builds one model chunk per rank, "
                "so a virtual pipeline degree of "
                f"{config.virtual_pipeline_model_parallel_size} is refused; "
                "the parameter check compares a whole stage"
            )
        if vp_stage is not None:
            raise ValueError(
                f"the stock megatron driver was asked for virtual pipeline "
                f"stage {vp_stage}; it builds one chunk per rank"
            )
        model = super().build_model(
            pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )
        args = get_args()
        shape = shape_by_name(args.bench_model_size)
        stages = args.pipeline_model_parallel_size
        stage = mpu.get_pipeline_model_parallel_rank()
        counted = sum(parameter.numel() for parameter in model.parameters())
        # The expert degree divides the routed experts a rank holds, so the
        # guard has to know it or it refuses an honest expert-parallel run.
        # It is read from the arguments rather than from ``mpu`` because
        # ``install_data_parallel_marker`` separately reads the BUILT expert
        # group and raises when the two disagree. So the argument is checked
        # against the world by that shim, and used here.
        expected = shape.stage_param_count(
            pipeline_degree=stages,
            stage_index=stage,
            expert_degree=args.expert_model_parallel_size,
        )
        if counted != expected:
            raise ValueError(
                f"stage {stage} of {stages} built {counted} parameters, "
                f"where shape {shape.name!r} declares {expected} at expert "
                f"degree {args.expert_model_parallel_size}; the model on "
                "this rank is not the model the run claims to measure"
            )
        print(
            STAGE_SIZE_LINE.format(
                stage=stage, stages=stages, count=counted
            ),
            flush=True,
        )
        print(
            MODEL_SIZE_LINE.format(
                size=shape.name, total=f"{shape.param_count:,}"
            ),
            flush=True,
        )
        return model
