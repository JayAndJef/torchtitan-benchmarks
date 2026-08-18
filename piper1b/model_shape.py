"""The model shapes both engines are built from.

Single source of truth for the ``--model-size`` run axis. ``piper1b/
config_registry.py`` builds the TorchTitan model from a ``PiperShape`` and
``megatron_baseline/model.py`` builds the megatron-core twin from the same
object, so the two engines cannot drift: a size is defined once here.

Deliberately imports nothing but ``dataclasses``. ``benchmarks/``,
``megatron_baseline/`` and ``tools/`` all import this module, several of them
in processes that must not pull in torch or torchtitan.

Two shapes are registered:

``normal``
    The historical piper-1B model: dim 1024, 16 layers, 1,066,241,024
    parameters. Every number this repo published before schema 9 is this
    shape.

``huge``
    One transformer layer at a much larger width, sized to fill an H200 (see
    ``reports/``'s memory-ceiling ladder). It exists to make the cuda-graph
    comparison be about a transformer block rather than about the embedding
    table: embedding + lm_head are ``2*V*D`` parameters and one layer is
    ``45*D^2``, so their ratio is ``2*151936/(45*D) = 6753/D``. At dim 1024 a
    1-layer model would be 87% embedding; at dim 12288 the single layer is
    64% of the parameters and the large majority of the FLOPs.

Everything that varies per shape is a field or a derived property of
``PiperShape``, so registering a new shape is one ``PIPER_SHAPES`` entry and
never an edit somewhere else. That includes the two knobs that used to be
hardcoded per size elsewhere: ``supports_block_regions`` (derived from
``n_layers``) and ``parity_gate`` (a field, read by
``tools/megatron_parity_check.py``).

A shape reaches TorchTitan as ``--config-arg size=<name>``, which
``benchmarks/runtime.py`` appends to the training command and the fork's
``ConfigManager`` forwards as a keyword argument to the ``--config``
function; ``piper1b/config_registry.py`` resolves it back through
``shape_by_name``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PiperShape:
    """One model geometry, shared by the TorchTitan and Megatron builders."""

    name: str
    dim: int
    n_layers: int
    head_dim: int = 64
    vocab_size: int = 151936
    num_experts: int = 4
    top_k: int = 2
    rope_theta: float = 1_000_000.0
    max_seq_len: int = 2048
    # Logit rel_l2 ceiling for tools/megatron_parity_check.py. bf16-scaled:
    # rel_l2 grows roughly with the square root of the reduction length, so a
    # wider shape legitimately needs a wider gate. Never widen one without the
    # --fp32-reference evidence that check's docstring describes.
    parity_gate: float = 2e-2

    def __post_init__(self) -> None:
        if self.dim % (2 * self.head_dim):
            raise ValueError(
                f"{self.name}: dim {self.dim} must be a multiple of "
                f"2*head_dim ({2 * self.head_dim}) so n_kv_heads is integral"
            )
        if self.dim % 2:
            raise ValueError(
                f"{self.name}: dim {self.dim} must be even so the 3.5x MoE "
                "hidden width is integral"
            )
        if self.n_layers < 1:
            raise ValueError(f"{self.name}: n_layers must be >= 1")

    # --- derived geometry -------------------------------------------------

    @property
    def supports_block_regions(self) -> bool:
        """False at one layer, where the block graph cannot be identified.

        ``benchmarks/profile_regions.py`` identifies a compiled block graph by
        its invocation count, ``n_layers * profiler_active``. At
        ``n_layers == 1`` that equals ``profiler_active``, which the loss- and
        embedding-side partitions also produce, so the match is ambiguous and
        the run declares no regions at all -- exactly as the megatron scenario
        does, and for the same reason.
        """
        return self.n_layers > 1

    @property
    def n_heads(self) -> int:
        return self.dim // self.head_dim

    @property
    def n_kv_heads(self) -> int:
        return self.n_heads // 2

    @property
    def heads_per_group(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def moe_hidden_dim(self) -> int:
        """Expert width: 3.5x dim (3584 at dim 1024, piper's inter_dim)."""
        return self.dim * 7 // 2

    @property
    def qkv_out_features(self) -> int:
        return (self.n_heads + 2 * self.n_kv_heads) * self.head_dim

    # --- parameter counting ----------------------------------------------
    #
    # Mirrors torchtitan's get_moe_model_nparams_and_flops
    # (third_party/torchtitan/torchtitan/models/utils.py), verified against
    # the "Total parameter count: dense D, sparse S, vision 0, active A" line
    # a real run logs. tests/test_model_shape.py pins the normal-size values.

    @property
    def _per_layer_dense(self) -> int:
        # attention_norm (D) + qkv (2*D*D) + q_norm/k_norm (2*head_dim)
        # + wo (D*D) + ffn_norm (D)
        return 3 * self.dim**2 + 2 * self.dim + 2 * self.head_dim

    @property
    def _router(self) -> int:
        return self.num_experts * self.dim

    @property
    def _experts(self) -> int:
        return 3 * self.num_experts * self.moe_hidden_dim * self.dim

    @property
    def _experts_active(self) -> int:
        return 3 * self.top_k * self.moe_hidden_dim * self.dim

    @property
    def nparams_embedding(self) -> int:
        return self.vocab_size * self.dim

    @property
    def nparams_dense(self) -> int:
        """Embedding + lm_head + final norm + every per-layer dense tensor."""
        return (
            2 * self.vocab_size * self.dim
            + self.dim
            + self.n_layers * self._per_layer_dense
        )

    @property
    def nparams_sparse(self) -> int:
        return self.n_layers * (self._router + self._experts)

    @property
    def nparams_active(self) -> int:
        return self.nparams_dense + self.n_layers * (
            self._router + self._experts_active
        )

    @property
    def param_count(self) -> int:
        return self.nparams_dense + self.nparams_sparse

    def num_flops_per_token(self, seq_len: int) -> int:
        """The tflops/MFU denominator both engines report against.

        ``head_dims`` in torchtitan's helper is qk + v = ``2*head_dim``.
        """
        nparams_for_matmul = self.nparams_active - self.nparams_embedding
        return (
            6 * nparams_for_matmul
            + 6 * self.n_layers * self.n_heads * (2 * self.head_dim) * seq_len
        )

    def describe(self, *, seq_len: int) -> dict[str, object]:
        """Flat JSON-safe provenance record for the manifest."""
        return {
            "name": self.name,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "moe_hidden_dim": self.moe_hidden_dim,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "vocab_size": self.vocab_size,
            "rope_theta": self.rope_theta,
            "max_seq_len": self.max_seq_len,
            "supports_block_regions": self.supports_block_regions,
            "parity_gate": self.parity_gate,
            "param_count": self.param_count,
            "nparams_dense": self.nparams_dense,
            "nparams_sparse": self.nparams_sparse,
            "nparams_active": self.nparams_active,
            "num_flops_per_token": self.num_flops_per_token(seq_len),
            "num_flops_per_token_seq_len": seq_len,
        }


NORMAL = PiperShape(name="normal", dim=1024, n_layers=16)

# dim chosen by the memory-ceiling probe: the largest multiple of 128 whose
# megatron-under-cuda-graph peak stays under ~125 GiB on a 139.81 GiB H200
# (megatron's graph mode allocates a bf16 main_grad for every parameter, so
# it costs 10 B/param of state against titan's 8 and sets the ceiling).
HUGE = PiperShape(
    name="huge",
    dim=12288,
    n_layers=1,
    # 12x the normal dim, so the bf16 accumulation error grows with it: the
    # normal shape's measured 5.5e-3 becomes 2.03e-2 (sqrt(12) = 3.46). This
    # is evidenced, not slack -- --fp32-reference puts titan's own bf16 output
    # 3.25e-2 from fp32 against megatron's 3.29e-2, so the engines agree with
    # each other better than either agrees with fp32.
    parity_gate=5e-2,
)

PIPER_SHAPES: dict[str, PiperShape] = {
    shape.name: shape for shape in (NORMAL, HUGE)
}


def shape_by_name(name: str) -> PiperShape:
    try:
        return PIPER_SHAPES[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown model size {name!r}. Available: "
            + ", ".join(PIPER_SHAPES)
        ) from error
