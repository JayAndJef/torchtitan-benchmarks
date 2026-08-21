"""The model shapes both engines are built from.

Single source of truth for the ``--model-size`` run axis.
``benchmarks/models/piper_qwen3/config_registry.py`` builds the TorchTitan
model from a ``PiperShape`` and ``benchmarks/models/piper_qwen3/
megatron_model.py`` builds the megatron-core twin from the same object, so
the two engines cannot drift: a size is defined once here.

Deliberately imports nothing but ``dataclasses``. Modules across
``benchmarks/`` and ``tools/`` import this module, several of them in
processes that must not pull in torch or torchtitan.

Four shapes are registered. The order, smallest to largest by parameter
count, is ``1b`` < ``large`` < ``huge`` < ``giant``. The names do not
carry that order on their own, so read it here: ``giant`` is above ``huge``.
``PIPER_SHAPES`` lists the shapes in that same order, and
``tests/test_model_shape.py`` asserts both the order and the ascending
parameter counts.

``normal`` is the retired name of ``1b`` and still resolves to it. See
``MODEL_SIZE_ALIASES`` below for why it cannot simply be deleted.

The one ratio every entry below refers to: embedding + lm_head are ``2*V*D``
parameters and one transformer layer is ``45*D^2``, so one layer against the
two tables is ``2*151936/(45*D) = 6753/D``. The whole layer stack against
them is ``n_layers*dim/6753``, which is why the layer count is part of a
shape and not a free choice.

``1b``
    Piper 1B: dim 1024, 16 layers, 1,066,241,024 parameters. Every number
    this repo published before schema 9 is this shape, under its old name
    ``normal``.

``large``
    Four transformer layers at dim 4096, 4,264,661,504 parameters. It is the
    middle rung between ``1b`` and ``huge`` in dim and in parameter
    count. The layer count is what makes it a rung of the same model rather
    than a different experiment: ``n_layers*dim`` is 16384 here, the product ``1b``
    carries, so ``large`` splits its parameters exactly as ``1b`` does --
    29% embedding tables, 71% layer stack. At one layer the ratio above is
    1.65, so a 1-layer model at dim 4096 would be 62% embedding table and the
    benchmark would measure the lm_head and the cross entropy. Four layers
    also keep ``supports_block_regions`` True, so ``large`` is the only shape
    above ``1b`` that validation rule 7 still guards.

``huge``
    One transformer layer at a much larger width, sized to fill an H200 (see
    ``reports/``'s memory-ceiling ladder). It exists to make the cuda-graph
    comparison be about a transformer block rather than about the embedding
    table. At dim 1024 a 1-layer model would be 87% embedding; at dim 12288
    the single layer is 64% of the parameters and the large majority of the
    FLOPs. It is the one shape whose ``n_layers*dim`` is not 16384, because
    the memory ceiling chose its dim rather than the parameter split.

``giant``
    One transformer layer at dim 16384, 17,058,349,184 parameters. The ratio
    above is 0.41 here, so the ``huge`` argument holds with room to spare,
    and ``n_layers*dim`` is 16384 again, so ``giant`` carries the ``1b``
    parameter split.

    This shape is declared from estimates. No scenario has run at it, in
    either system, and the first run can run out of memory. The ``GIANT``
    constant records the memory arithmetic and what it does not cover.

Everything that varies per shape is a field or a derived property of
``PiperShape``, so registering a new shape is one ``PIPER_SHAPES`` entry and
never an edit somewhere else. That includes the two knobs that used to be
hardcoded per size elsewhere: ``supports_block_regions`` (derived from
``n_layers``) and ``parity_gate`` (a field, read by
``tools/megatron_parity_check.py``).

A shape reaches TorchTitan as ``--config-arg size=<name>``, which
``benchmarks/e2e/launch.py`` appends to the training command and the fork's
``ConfigManager`` forwards as a keyword argument to the ``--config``
function; ``benchmarks/models/piper_qwen3/config_registry.py`` resolves it
back through ``shape_by_name``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PiperShape:
    """One model geometry, shared by the TorchTitan and Megatron builders.

    A number is a field when the registered shapes disagree about it. Where
    they all agree, it is a default here or a derivation below. That rule is
    what keeps one number from having two sources in this file.

    ``head_dim``, ``n_kv_heads`` and ``num_experts`` are fields for that
    reason. Each was a constant or a derivation until the real piper ladder
    arrived, and each is wrong there. Piper 48B carries ``head_dim`` 128
    against 1B's 64. Piper 9B and 48B carry 8 kv heads, where the old rule
    ``n_kv_heads = n_heads // 2`` returns 16 and 16. Piper 9B and 48B carry 8
    experts against 1B's 4. Each wrong value builds a different model and
    publishes it under the requested name.

    ``PiperShape.derived`` applies the piper-1B family rules for a probe or a
    test shape. Never register a shape through it.
    """

    name: str
    dim: int
    n_layers: int
    head_dim: int
    n_kv_heads: int
    num_experts: int
    # The registered shapes agree on these four, so each is one default here
    # rather than a value repeated per shape. Promote one to a per-shape value
    # the moment a registered shape disagrees, and not before.
    top_k: int = 2
    vocab_size: int = 151936
    rope_theta: float = 1_000_000.0
    max_seq_len: int = 2048
    # Logit rel_l2 ceiling for tools/megatron_parity_check.py. bf16-scaled:
    # rel_l2 grows roughly with the square root of the reduction length, so a
    # wider shape legitimately needs a wider gate. Never widen one without the
    # --fp32-reference evidence that check's docstring describes.
    parity_gate: float = 2e-2

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError(f"{self.name}: dim must be >= 1")
        if self.n_layers < 1:
            raise ValueError(f"{self.name}: n_layers must be >= 1")
        if self.head_dim < 1:
            raise ValueError(f"{self.name}: head_dim must be >= 1")
        if self.dim % self.head_dim:
            raise ValueError(
                f"{self.name}: dim {self.dim} must be a multiple of head_dim "
                f"{self.head_dim}, because n_heads is derived from the two"
            )
        if self.dim % 2:
            raise ValueError(
                f"{self.name}: dim {self.dim} must be even so the 3.5x MoE "
                "hidden width is integral"
            )
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
        """Build a probe shape from the piper-1B family rules.

        The rules are ``head_dim`` 64, ``n_kv_heads = n_heads // 2`` and 4
        experts. They describe piper 1B and no other real piper model: 9B and
        48B carry 8 kv heads and 8 experts, and 48B carries ``head_dim`` 128.

        Use this for a test or a probe shape, whose exact geometry is
        arbitrary and which no run publishes. **Never register a shape through
        it.** Every ``PIPER_SHAPES`` entry writes its geometry out, so a
        reader can check the entry against the model config it claims to be.
        """
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

    # --- derived geometry -------------------------------------------------

    @property
    def supports_block_regions(self) -> bool:
        """False at one layer, where the block graph cannot be identified.

        ``benchmarks/traces/extraction.py`` identifies a compiled block
        graph by its invocation count, ``n_layers * profiler_active``. At
        ``n_layers == 1`` that equals ``profiler_active``, which the loss- and
        embedding-side partitions also produce, so the match is ambiguous and
        the run declares no regions at all -- exactly as the megatron scenario
        does, and for the same reason.
        """
        return self.n_layers > 1

    @property
    def n_heads(self) -> int:
        """Query heads: ``dim // head_dim``.

        Derived because every registered shape satisfies
        ``n_heads * head_dim == dim``, and ``__post_init__`` enforces the
        divisibility that makes it exact. It is not universal -- Qwen3
        30B-A3B carries 32 heads of 128 at dim 2048 -- so promote it to a
        field if such a shape is ever registered.
        """
        return self.dim // self.head_dim

    @property
    def heads_per_group(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def moe_hidden_dim(self) -> int:
        """Expert width: 3.5x dim (3584 at dim 1024, piper's inter_dim).

        Derived because every registered shape agrees with it: piper 1B 3584,
        9B 7168 and 48B 14336 are each 3.5x their dim, and the three synthetic
        shapes were built on the same rule. Piper 9M is the one real config
        that breaks it (128 against 896), so registering 9M means promoting
        this to a field.
        """
        return self.dim * 7 // 2

    @property
    def qkv_out_features(self) -> int:
        return (self.n_heads + 2 * self.n_kv_heads) * self.head_dim

    # --- parameter counting ----------------------------------------------
    #
    # Mirrors torchtitan's get_moe_model_nparams_and_flops
    # (third_party/torchtitan/torchtitan/models/utils.py), verified against
    # the "Total parameter count: dense D, sparse S, vision 0, active A" line
    # a real run logs. tests/test_model_shape.py pins the 1b values.

    @property
    def _per_layer_dense(self) -> int:
        """attention_norm + fused qkv + q_norm/k_norm + wo + ffn_norm.

        Written from the tensor widths rather than as ``3*D^2 + 2*D +
        2*head_dim``. That closed form assumes ``qkv_out_features == 2*dim``
        and ``n_heads*head_dim == dim``, which hold only at a 2:1 query-to-kv
        ratio. Piper 9B and 48B run 4:1, where the fused qkv is
        ``1.5*dim`` wide, so the closed form overcounts them. The two forms
        agree exactly on every 2:1 shape.
        """
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


# Piper 1B, verbatim. Every field matches
# /data/zejiaqi/piper/examples/models/qwen3.py case '1B'.
PIPER_1B = PiperShape(
    name="1b",
    dim=1024,
    n_layers=16,
    head_dim=64,
    n_kv_heads=8,
    num_experts=4,
)

# dim chosen by the memory-ceiling probe: the largest multiple of 128 whose
# megatron-under-cuda-graph peak stays under ~125 GiB on a 139.81 GiB H200
# (megatron's graph mode allocates a bf16 main_grad for every parameter, so
# it costs 10 B/param of state against titan's 8 and sets the ceiling).
HUGE = PiperShape(
    name="huge",
    dim=12288,
    n_layers=1,
    head_dim=64,
    # 2:1 query-to-kv, the piper-1B ratio this shape was built on. Real piper
    # runs 4:1 from 9B up, so 96 kv heads is not a piper geometry.
    n_kv_heads=96,
    num_experts=4,
    # 12x the 1b dim, so the bf16 accumulation error grows with it: the
    # 1b shape's measured 5.5e-3 becomes 2.03e-2 (sqrt(12) = 3.46). This
    # is evidenced, not slack -- --fp32-reference puts titan's own bf16 output
    # 3.25e-2 from fp32 against megatron's 3.29e-2, so the engines agree with
    # each other better than either agrees with fp32.
    parity_gate=5e-2,
)

# The middle rung: between 1b and huge in dim and in parameter count.
# n_layers*dim is
# 16384, the product the 1b shape carries, so large reproduces the 1b
# shape's parameter split exactly: 29% embedding tables, 71% layer stack. The
# layer count is what buys that. At dim 4096 the 6753/D ratio is 1.65, so a
# 1-layer model here would be 62% embedding table, and the benchmark would
# measure the lm_head and the cross entropy rather than a transformer block.
# Four layers also keep supports_block_regions True.
#
# No scenario has run at this shape, in either system.
LARGE = PiperShape(
    name="large",
    dim=4096,
    n_layers=4,
    head_dim=64,
    n_kv_heads=32,
    num_experts=4,
    # UNVERIFIED. Nothing has measured this shape's logit rel_l2. The two
    # measured shapes fit rel_l2 = 5.5e-3 * sqrt(dim/1024) to within 7%
    # (1b 5.5e-3 at dim 1024; huge 2.03e-2 at dim 12288 against 1.9e-2
    # predicted), which puts dim 4096 near 1.2e-2. 3e-2 keeps the 2.46x margin
    # huge keeps over its own measurement. Two anchors cannot separate the
    # width term from a depth term -- 1b is 16 layers at dim 1024 and huge
    # is 1 layer at dim 12288 -- and this shape is 4 layers, so the estimate
    # is weaker here than the number alone suggests. Run
    # tools/megatron_parity_check.py --model-size large before any parity
    # claim, and --fp32-reference before you change this value.
    parity_gate=3e-2,
)

# Above huge, and declared from arithmetic alone. n_layers*dim is 16384, so
# giant carries the 1b shape's parameter split, and the 6753/D ratio is
# 0.41, so huge's one-layer argument holds with room to spare.
#
# NOTHING HAS MEASURED THIS SHAPE. No e2e scenario and no kernel scenario has
# run at it, and the first run can run out of memory. The binding case is the
# expert_mlp kernel scenario, whose expert weights are
# 3 * num_experts * moe_hidden_dim * dim elements: 21.0 GiB per arm in bf16,
# 42.0 GiB once the weight gradient grows, against a 139.81 GiB H200. The
# correctness gate adds an fp32 copy of the same three tensors, 42.0 GiB,
# resident for the whole pass. That is 84 GiB before any activation. It fits
# one arm at a time -- run_correctness_pass builds one arm at a time and drops
# it before the next -- but swiglu already ran out of memory at huge once
# (out/20260819T010252Z/kernels/), at 56% of these tensor sizes and before
# that residency change. The estimate covers weights and their gradients only;
# it covers no activation, no workspace and no allocator fragmentation.
# Measure before you report anything about this shape.
GIANT = PiperShape(
    name="giant",
    dim=16384,
    n_layers=1,
    head_dim=64,
    n_kv_heads=128,
    num_experts=4,
    # UNVERIFIED, on the same sqrt(dim) law LARGE uses: dim 16384 predicts
    # about 2.4e-2, and 6e-2 keeps huge's 2.46x margin over it. The depth
    # caveat on LARGE does not apply here, because giant is one layer as huge
    # is, so the width extrapolation is the only step. Run
    # tools/megatron_parity_check.py --model-size giant before any parity
    # claim, and --fp32-reference before you change this value.
    parity_gate=6e-2,
)

# Declared smallest to largest by parameter count. The names do not carry the
# order, so the declaration does.
PIPER_SHAPES: dict[str, PiperShape] = {
    shape.name: shape for shape in (PIPER_1B, LARGE, HUGE, GIANT)
}

# Retired ``--model-size`` names, each mapped to the key that replaced it.
#
# ``normal`` was the 1B shape's name until it took the real model's name. It
# stays accepted, and it must: 13 manifests under ``out/`` record
# ``"model_size": "normal"``, every manifest at schema <= 8 carries no
# ``model_size`` at all and is defined to resume as that shape, and
# ``--resume`` compares the recorded string against the requested one. A bare
# rename would refuse a resume that should succeed.
#
# Aliases live here and not in ``PIPER_SHAPES``, so that ``PIPER_SHAPES``
# enumerates the shapes and nothing else. The CLI's choice lists and the
# tests both derive from it, and a shape counted twice would appear twice.
MODEL_SIZE_ALIASES: dict[str, str] = {"normal": "1b"}

# Every ``--model-size`` value a command accepts: the shapes, then the
# aliases.
MODEL_SIZE_CHOICES: tuple[str, ...] = tuple(PIPER_SHAPES) + tuple(
    MODEL_SIZE_ALIASES
)


def canonical_size_name(name: str) -> str:
    """The registry key a ``--model-size`` value names.

    Total on purpose: an unknown name passes through unchanged. Callers that
    compare two recorded values -- the resume predicate is the one that
    matters -- must normalise both sides, and a manifest can record any
    string at all. Rejecting an unknown size is ``shape_by_name``'s job.
    """
    return MODEL_SIZE_ALIASES.get(name, name)


def shape_by_name(name: str) -> PiperShape:
    try:
        return PIPER_SHAPES[canonical_size_name(name)]
    except KeyError as error:
        raise ValueError(
            f"Unknown model size {name!r}. Available: "
            + ", ".join(MODEL_SIZE_CHOICES)
        ) from error
