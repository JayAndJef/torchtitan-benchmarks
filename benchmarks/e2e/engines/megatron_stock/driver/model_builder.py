"""The stock GPT model builder, plus a check of the parameter count of each stage and the line that the model fact reads.

This module imports megatron, so only the training process imports it, through ``BenchGPTModelConfig.builder``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from megatron.core import mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training import get_args
from megatron.training.models.gpt import GPTModelBuilder, GPTModelConfig

from benchmarks.e2e.engines.megatron_stock.driver.markers import (
    MODEL_SIZE_LINE,
    STAGE_SIZE_LINE,
)
from benchmarks.models.piper_qwen3.shape import shape_by_name


@dataclass(kw_only=True)
class BenchGPTModelConfig(GPTModelConfig):
    """``GPTModelConfig`` with ``CountingGPTModelBuilder`` as its builder."""

    builder: ClassVar[str] = (
        "benchmarks.e2e.engines.megatron_stock.driver.model_builder.CountingGPTModelBuilder"
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
        """Build one stage, check its parameter count against the shape, and print the two parameter lines."""
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
