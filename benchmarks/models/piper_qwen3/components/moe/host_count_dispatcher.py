"""The all-to-all token dispatcher with the rows of each local expert on the host, as a TorchTitan override.

The stock dispatcher already makes one blocking copy per layer and microbatch: the row sums of the
(ep, e) count matrix. This one copies the whole matrix in that copy, so the experts get their counts
with no new sync.

Activation:
    --override.imports benchmarks.models.piper_qwen3.components.moe.host_count_dispatcher.host_count_dispatcher
"""

from dataclasses import dataclass

import torch

from torchtitan.config import derive, override
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


@torch.library.custom_op(
    "engine_bench::host_token_counts", mutates_args=(), device_types="cuda"
)
def host_token_counts(
    local_E: torch.Tensor, global_EP_e: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The input splits, the output splits and the rows of each local expert, as int64 host tensors."""
    ep, e = global_EP_e.shape
    if local_E.ndim != 1 or local_E.shape[0] != ep * e:
        raise RuntimeError(
            f"host_token_counts got local counts {tuple(local_E.shape)} for a "
            f"({ep}, {e}) count matrix"
        )
    if local_E.dtype != torch.int64 or global_EP_e.dtype != torch.int64:
        raise RuntimeError(
            f"host_token_counts needs int64 counts, got {local_E.dtype} and "
            f"{global_EP_e.dtype}"
        )
    input_splits = (
        local_E.view(ep, e).sum(dim=1).to(torch.device("cpu"), non_blocking=True)
    )
    # The one blocking copy; it also completes the copy of the input splits.
    host_EP_e = global_EP_e.to(torch.device("cpu"), non_blocking=False)
    return input_splits, host_EP_e.sum(dim=1), host_EP_e.sum(dim=0)


@host_token_counts.register_fake
def _host_token_counts_fake(local_E, global_EP_e):
    ep, e = global_EP_e.shape
    host = torch.device("cpu")
    return (
        torch.empty((ep,), dtype=torch.int64, device=host),
        torch.empty((ep,), dtype=torch.int64, device=host),
        torch.empty((e,), dtype=torch.int64, device=host),
    )


class HostCountDispatcher(AllToAllTokenDispatcher):
    """AllToAllTokenDispatcher that returns the rows of each local expert as an int64 host tensor."""

    @dataclass(kw_only=True, slots=True)
    class Config(AllToAllTokenDispatcher.Config):
        pass

    def __init__(self, config: "HostCountDispatcher.Config"):
        super().__init__(config)
        # Rows per local expert, (e,) int64 on the host: the hook sets it, dispatch takes it.
        self._host_rows_e: torch.Tensor | None = None

    # pyrefly: ignore [bad-override]
    def dispatch(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ):
        if self.ep_mesh is None:
            raise RuntimeError(
                "HostCountDispatcher needs an expert-parallel mesh; at ep 1 "
                "the stock dispatcher keeps the counts on the device"
            )
        if get_spmd_backend() == "spmd_types":
            raise RuntimeError(
                "HostCountDispatcher has no SPMD type rule for its custom op"
            )
        # Runs the hook _sync_token_count_exchange below, which sets _host_rows_e.
        routed_input_RD, _, metadata = super().dispatch(
            x_TD, topk_scores_TK, topk_expert_ids_TK, num_local_tokens_per_expert_E
        )
        host_rows_e, self._host_rows_e = self._host_rows_e, None
        if host_rows_e is None:
            raise RuntimeError("HostCountDispatcher.dispatch made no count exchange")
        return routed_input_RD, host_rows_e, metadata

    def _sync_token_count_exchange(
        self,
        num_local_tokens_per_expert_E: torch.Tensor,
        num_global_tokens_per_local_expert_EP_e: torch.Tensor,
        ep_size: int,
    ) -> tuple[torch.Tensor, list[int], list[int]]:
        """The stock ``dispatch`` calls this hook once; it also keeps the rows of each local expert in ``_host_rows_e``."""
        num_global_tokens_per_local_expert_EP_e = (
            torch.ops._c10d_functional.wait_tensor(
                num_global_tokens_per_local_expert_EP_e
            )
        )
        input_splits, output_splits, self._host_rows_e = host_token_counts(
            num_local_tokens_per_expert_E,
            num_global_tokens_per_local_expert_EP_e.view(ep_size, -1),
        )
        return (
            num_global_tokens_per_local_expert_EP_e.reshape(-1),
            input_splits.tolist(),
            output_splits.tolist(),
        )


@override(
    target=AllToAllTokenDispatcher.Config,
    exact=True,
    description="All-to-all dispatch that returns the rows of each local expert on the host.",
)
def host_count_dispatcher(
    cfg: AllToAllTokenDispatcher.Config,
) -> HostCountDispatcher.Config:
    return derive(cfg, HostCountDispatcher.Config)
