"""The model shapes that both engines build."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PiperShape:
    """One model geometry, which the TorchTitan and the Megatron builders share."""

    name: str
    dim: int
    n_layers: int
    head_dim: int
    n_kv_heads: int
    num_experts: int
    n_heads: int | None = None
    """The query heads; ``None`` derives ``dim // head_dim``, and a registered shape can differ from that quotient."""
    moe_hidden_dim: int | None = None
    """The expert width; ``None`` derives ``3.5 * dim``."""
    top_k: int = 2
    """The routed experts per token."""
    vocab_size: int = 151936
    rope_theta: float = 1_000_000.0
    max_seq_len: int = 4096
    """The size of the RoPE cache and the sequence ceiling; a longer sequence fails the eager bounds check and reads out of bounds under compile."""
    parity_gate: float = 2e-2
    """The rel_l2 ceiling of a cross-engine logit comparison; no check gates on it, and only ``1b`` and ``huge`` carry a measured value."""

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"{self.name}: dim must be >= 1")
        if self.n_layers < 1:
            raise ValueError(f"{self.name}: n_layers must be >= 1")
        if self.head_dim < 1:
            raise ValueError(f"{self.name}: head_dim must be >= 1")
        # Each divisibility guard applies to the derivation alone.
        if self.n_heads is None:
            if self.dim % self.head_dim:
                raise ValueError(
                    f"{self.name}: dim {self.dim} must be a multiple of "
                    f"head_dim {self.head_dim}, because n_heads is derived "
                    "from the two; pass n_heads for a shape whose heads do "
                    "not tile dim"
                )
            object.__setattr__(self, "n_heads", self.dim // self.head_dim)
        if self.moe_hidden_dim is None:
            if self.dim % 2:
                raise ValueError(
                    f"{self.name}: dim {self.dim} must be even so the 3.5x "
                    "MoE hidden width is integral; pass moe_hidden_dim for a "
                    "shape with its own expert width"
                )
            object.__setattr__(self, "moe_hidden_dim", self.dim * 7 // 2)
        if self.n_heads < 1:
            raise ValueError(f"{self.name}: n_heads must be >= 1")
        if self.moe_hidden_dim < 1:
            raise ValueError(f"{self.name}: moe_hidden_dim must be >= 1")
        if self.n_kv_heads < 1:
            raise ValueError(f"{self.name}: n_kv_heads must be >= 1")
        if self.n_heads % self.n_kv_heads:
            raise ValueError(
                f"{self.name}: n_heads {self.n_heads} must be a multiple of "
                f"n_kv_heads {self.n_kv_heads}, so each kv head serves a "
                "whole query group"
            )
        if self.num_experts < 1:
            raise ValueError(f"{self.name}: num_experts must be >= 1")

    @classmethod
    def derived(
        cls,
        *,
        name: str,
        dim: int,
        n_layers: int,
        head_dim: int = 64,
        n_kv_heads: int | None = None,
        num_experts: int = 4,
        **rest: object,
    ) -> "PiperShape":
        """A probe shape from the piper-1B family rules; never register a shape through it."""
        if dim % head_dim:
            raise ValueError(
                f"{name}: dim {dim} must be a multiple of head_dim "
                f"{head_dim} for the derived n_heads to be integral"
            )
        n_heads = dim // head_dim
        if n_kv_heads is None:
            if n_heads % 2:
                raise ValueError(
                    f"{name}: n_heads {n_heads} is odd, so the family rule "
                    "n_kv_heads = n_heads // 2 does not apply; pass n_kv_heads"
                )
            n_kv_heads = n_heads // 2
        return cls(
            name=name,
            dim=dim,
            n_layers=n_layers,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
            num_experts=num_experts,
            **rest,  # type: ignore[arg-type]
        )

    @property
    def heads_per_group(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def qkv_out_features(self) -> int:
        return (self.n_heads + 2 * self.n_kv_heads) * self.head_dim

    @property
    def _per_layer_dense(self) -> int:
        """The dense parameters of one layer: the attention norm, the fused qkv, the q and k norms, ``wo`` and the ffn norm."""
        return (
            self.dim * self.qkv_out_features
            + self.n_heads * self.head_dim * self.dim
            + 2 * self.dim
            + 2 * self.head_dim
        )

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
        """The embedding, the output head, the final norm and the dense parameters of every layer."""
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
        """Every parameter of the model, dense and sparse; the parameter count line of a run checks it."""
        return self.nparams_dense + self.nparams_sparse

    @property
    def _per_layer(self) -> int:
        """Every parameter of one transformer layer, dense and sparse."""
        return self._per_layer_dense + self._router + self._experts

    def stage_param_count(
        self,
        *,
        pipeline_degree: int,
        stage_index: int,
        expert_degree: int = 1,
    ) -> int:
        """The parameters that one rank holds at pipeline stage ``stage_index``, with the routed experts split over ``expert_degree`` ranks.

        Each stage holds ``n_layers // pipeline_degree`` layers. The first
        stage also holds the embedding, and the last stage also holds the
        final norm and the output head. The router and the dense parameters
        do not divide by the expert degree.
        """
        if pipeline_degree < 1:
            raise ValueError(
                f"{self.name}: pipeline_degree {pipeline_degree} must be >= 1"
            )
        if not 0 <= stage_index < pipeline_degree:
            raise ValueError(
                f"{self.name}: stage_index {stage_index} is outside the "
                f"{pipeline_degree} stage(s) of this pipeline"
            )
        if self.n_layers % pipeline_degree:
            raise ValueError(
                f"{self.name}: {self.n_layers} layers do not divide evenly "
                f"into {pipeline_degree} pipeline stages"
            )
        if expert_degree < 1:
            raise ValueError(
                f"{self.name}: expert_degree {expert_degree} must be >= 1"
            )
        if self.num_experts % expert_degree:
            raise ValueError(
                f"{self.name}: {self.num_experts} experts do not divide "
                f"evenly into {expert_degree} expert-parallel rank(s)"
            )
        per_layer = (
            self._per_layer_dense
            + self._router
            + self._experts // expert_degree
        )
        count = (self.n_layers // pipeline_degree) * per_layer
        if stage_index == 0:
            count += self.nparams_embedding
        if stage_index == pipeline_degree - 1:
            # The final norm and the untied output head.
            count += self.dim + self.vocab_size * self.dim
        return count

    def num_flops_per_token(self, seq_len: int) -> int:
        """The FLOPs per token that both engines divide by for their tflops and MFU figures."""
        nparams_for_matmul = self.nparams_active - self.nparams_embedding
        return (
            6 * nparams_for_matmul
            + 6 * self.n_layers * self.n_heads * (2 * self.head_dim) * seq_len
        )

    def describe(self, *, seq_len: int) -> dict[str, object]:
        """The JSON record of the shape that a manifest holds."""
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
            "parity_gate": self.parity_gate,
            "param_count": self.param_count,
            "nparams_dense": self.nparams_dense,
            "nparams_sparse": self.nparams_sparse,
            "nparams_active": self.nparams_active,
            "num_flops_per_token": self.num_flops_per_token(seq_len),
            "num_flops_per_token_seq_len": seq_len,
        }


PIPER_1B = PiperShape(
    name="1b",
    dim=1024,
    n_layers=16,
    head_dim=64,
    n_kv_heads=8,
    num_experts=4,
)
"""Piper 1B: dim 1024, 16 layers, 1,066,241,024 parameters."""

HUGE = PiperShape(
    name="huge",
    dim=12288,
    n_layers=1,
    head_dim=64,
    n_kv_heads=96,
    num_experts=4,
    parity_gate=5e-2,
)
"""One transformer layer at dim 12288, sized to fill an H200."""

LARGE = PiperShape(
    name="large",
    dim=4096,
    n_layers=4,
    head_dim=64,
    n_kv_heads=32,
    num_experts=4,
    parity_gate=3e-2,
)
"""Four transformer layers at dim 4096, 4,264,661,504 parameters, with the parameter split of ``1b``."""

GIANT = PiperShape(
    name="giant",
    dim=16384,
    n_layers=1,
    head_dim=64,
    n_kv_heads=128,
    num_experts=4,
    parity_gate=6e-2,
)
"""One transformer layer at dim 16384, 17,058,349,184 parameters; the ``expert_mlp`` kernel scenario can run out of memory at this shape."""

PIPER_9B = PiperShape(
    name="9b",
    dim=2048,
    n_layers=24,
    head_dim=64,
    n_kv_heads=8,
    num_experts=8,
    parity_gate=2e-2,
)
"""Piper 9B: dim 2048, 24 layers, 9,330,201,600 parameters, with 4:1 grouped-query attention and 8 experts."""

PIPER_30B_A3B = PiperShape(
    name="30b-a3b",
    dim=2048,
    n_layers=48,
    head_dim=128,
    n_kv_heads=4,
    num_experts=128,
    n_heads=32,
    moe_hidden_dim=768,
    top_k=8,
)
"""Qwen3-30B-A3B: 30,532,122,624 parameters, 3,353,032,704 of them active; it does not fit one H200.

Its ``max_seq_len`` is 4096, and Piper declares 262144. A run above
sequence length 4096 needs a wider RoPE cache.
"""

PIPER_30B_A3B_CUT = PiperShape(
    name="30b-a3b-20l",
    dim=2048,
    n_layers=20,
    head_dim=128,
    n_kv_heads=4,
    num_experts=128,
    n_heads=32,
    moe_hidden_dim=768,
    top_k=8,
)
"""Qwen3-30B-A3B cut to 20 layers: 13,084,744,704 parameters; it fits four H200 at dp 4 x ep 4 under ZeRO-1."""

PIPER_48B = PiperShape(
    name="48b",
    dim=4096,
    n_layers=32,
    head_dim=128,
    n_kv_heads=8,
    num_experts=8,
    parity_gate=3e-2,
)
"""Piper 48B: dim 4096, 32 layers, 47,685,316,608 parameters; its state needs about 355 GiB, so it does not fit one H200."""

PIPER_SHAPES: dict[str, PiperShape] = {
    shape.name: shape
    for shape in (
        PIPER_1B, LARGE, PIPER_9B, HUGE, PIPER_30B_A3B_CUT, GIANT, PIPER_30B_A3B,
        PIPER_48B
    )
}
"""Every registered shape, smallest to largest by parameter count."""

MODEL_SIZE_ALIASES: dict[str, str] = {"normal": "1b"}
"""The ``--model-size`` aliases, each mapped to the name of its shape."""

MODEL_SIZE_CHOICES: tuple[str, ...] = tuple(PIPER_SHAPES) + tuple(
    MODEL_SIZE_ALIASES
)
"""Every ``--model-size`` value a command accepts: shapes, then aliases."""


def canonical_size_name(name: str) -> str:
    """The shape name that a ``--model-size`` value names; an unknown name passes through unchanged."""
    return MODEL_SIZE_ALIASES.get(name, name)


def shape_by_name(name: str) -> PiperShape:
    """The registered shape that ``name`` names; an unknown name raises and names the choices."""
    try:
        return PIPER_SHAPES[canonical_size_name(name)]
    except KeyError as error:
        raise ValueError(
            f"Unknown model size {name!r}. Available: "
            + ", ".join(MODEL_SIZE_CHOICES)
        ) from error
