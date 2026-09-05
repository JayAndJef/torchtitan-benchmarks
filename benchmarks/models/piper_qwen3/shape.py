"""The model shapes both engines are built from.

Single source of truth for the ``--model-size`` run axis.
``benchmarks/models/piper_qwen3/config_registry.py`` builds the TorchTitan
model from a ``PiperShape`` and ``benchmarks/models/piper_qwen3/
megatron_model.py`` builds the megatron-core twin from the same object, so
the two engines cannot drift: a size is defined once here.

Deliberately imports nothing but ``dataclasses``. Modules across
``benchmarks/`` and ``tools/`` import this module, several of them in
processes that must not pull in torch or torchtitan.

Seven shapes are registered. The order, smallest to largest by parameter
count, is ``1b`` < ``large`` < ``9b`` < ``huge`` < ``giant`` < ``30b-a3b``
< ``48b``. The names do not carry that order on their own, so read it
here: ``giant`` is above ``huge``, and the real ladder interleaves with the
synthetic one. ``PIPER_SHAPES`` lists the shapes in that same order, and
``tests/test_model_shape.py`` asserts both the order and the ascending
parameter counts.

``normal`` is the retired name of ``1b`` and still resolves to it. See
``MODEL_SIZE_ALIASES`` below for why it cannot simply be deleted.

**Four of the seven are real piper models, and three are benchmark
inventions. The name says which.** ``1b``, ``9b``, ``30b-a3b`` and ``48b``
are transcribed field for field from ``examples/models/qwen3.py`` in the
piper checkout, cases ``'1B'``, ``'9B'``, ``'30B-A3B'`` and ``'48B'``. The
absolute path of that checkout moves with the host. It is
``/m-coriander/coriander/jayden/piper/`` today, and the
``/data/zejiaqi/piper/`` that ``config_registry.py`` cites is stale.

``large``, ``huge`` and ``giant`` are ours. Each was built by choosing a dim
and a layer count for a benchmark reason, and then applying the piper-1B
rules to everything else.

**A synthetic shape is not piper at scale, and ``huge`` is the clearest
case.** Piper scales by adding layers and experts and by widening
``head_dim``, and it holds ``n_heads`` at 32 and ``n_kv_heads`` at 8 from 9B
up -- 4:1 grouped-query attention. The synthetic shapes pin ``head_dim`` at
64 and take ``n_heads = dim/head_dim``, so their head count grows with the
width. ``huge`` carries 192 query heads over 96 kv heads, and ``giant``
carries 256 over 128. Real piper never approaches those counts. Their 2:1
ratio is also the ratio real piper stopped using after 1B. That changes the
fused qkv width, the attention arithmetic and the kv-cache size. Read
``huge`` as a wide transformer block sized to fill an H200, which is what it
was built to be, and never as piper at scale.

``large`` and ``48b`` make the same point at one dim. Both are dim 4096
with a 14336 expert width. They then differ in every remaining geometry
value: ``head_dim`` 64 against 128, ``n_heads`` 64 against 32,
``n_kv_heads`` 32 against 8, ``n_layers`` 4 against 32, ``num_experts`` 4
against 8. ``large`` does not approximate ``48b``.

The one ratio every entry below refers to: embedding + lm_head are ``2*V*D``
parameters and one transformer layer is ``45*D^2``, so one layer against the
two tables is ``2*151936/(45*D) = 6753/D``. The whole layer stack against
them is ``n_layers*dim/6753``, which is why the layer count is part of a
shape and not a free choice.

``1b`` (real)
    Piper 1B, verbatim: dim 1024, 16 layers, 1,066,241,024 parameters.
    Every number this repo published before schema 9 is this shape, under
    its old name ``normal``.

``large`` (synthetic)
    Four transformer layers at dim 4096, 4,264,661,504 parameters. It is the
    middle rung between ``1b`` and ``huge`` in dim and in parameter
    count. The layer count is what makes it a rung of the same model rather
    than a different experiment: ``n_layers*dim`` is 16384 here, the
    product ``1b`` carries, so ``large`` splits its parameters as ``1b``
    does --
    29% embedding tables, 71% layer stack. At one layer the ratio above is
    1.65, so a 1-layer model at dim 4096 would be 62% embedding table and the
    benchmark would measure the lm_head and the cross entropy. Four layers
    also keep ``supports_block_regions`` True, so ``large`` is the only shape
    above ``1b`` that validation rule 7 still guards.

``huge`` (synthetic)
    One transformer layer at a much larger width, sized to fill an H200 (see
    ``reports/``'s memory-ceiling ladder). It exists to make the cuda-graph
    comparison be about a transformer block rather than about the embedding
    table. At dim 1024 a 1-layer model would be 87% embedding; at dim 12288
    the single layer is 64% of the parameters and the large majority of the
    FLOPs. Its ``n_layers*dim`` is not 16384, because the memory ceiling
    chose its dim rather than the parameter split. It was the only such
    shape until the real ladder arrived; ``9b`` and ``48b`` do not hold that
    product either, because piper never chose it.

``giant`` (synthetic)
    One transformer layer at dim 16384, 17,058,349,184 parameters. The ratio
    above is 0.41 here, so the ``huge`` argument holds with room to spare,
    and ``n_layers*dim`` is 16384 again, so ``giant`` carries the ``1b``
    parameter split.

    This shape is declared from estimates. No scenario has run at it, in
    either system, and the first run can run out of memory. The ``GIANT``
    constant records the memory arithmetic and what it does not cover.

``9b`` (real)
    Piper 9B, verbatim: dim 2048, 24 layers, 9,330,201,600 parameters. The
    first registered shape with 4:1 grouped-query attention and 8 experts.

``30b-a3b`` (real)
    Qwen3-30B-A3B as piper's registry declares it, verbatim: dim 2048, 48
    layers, 32 heads of 128 over 4 kv heads, 128 experts of width 768 at
    top-8, 30,532,122,624 parameters of which 3,353,032,704 are active.
    It is the shape that made ``n_heads`` and ``moe_hidden_dim`` fields:
    its heads do not tile its dim, and its expert width is 768 against a
    derived 7168. Piper declares it and never ran it; nothing here has run
    it either, and no parity check has. See ``PIPER_30B_A3B`` for what its
    registration does and does not carry.

``48b`` (real)
    Piper 48B, verbatim: dim 4096, 32 layers, 47,685,316,608 parameters.
    The first registered shape with ``head_dim`` 128. Nothing has run at
    either. 9B may fit one H200 and 48B cannot; see the ``PIPER_48B``
    constant for that arithmetic.

Everything that varies per shape is a field or a derived property of
``PiperShape``, so registering a new shape is one ``PIPER_SHAPES`` entry and
never an edit somewhere else. That includes the two knobs that used to be
hardcoded per size elsewhere: ``supports_block_regions`` (derived from
``n_layers``) and ``parity_gate`` (a field, read by
``tools/megatron_parity_check.py``).

Registering one also means writing its numbers into
``tests/test_model_shape.py``'s ``PINNED_SHAPES``, which a test requires.
That is the second statement of the geometry, and it is deliberate: it is
transcribed from the model config the shape claims to be, so a derivation
that is wrong for that model cannot pass both.

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

    ``n_heads`` and ``moe_hidden_dim`` are fields with a derived default.
    Every shape registered before 2026-09-05 satisfies ``n_heads * head_dim
    == dim`` and ``moe_hidden_dim == dim * 7 // 2``, so ``None`` -- the
    default -- means "derive it from those rules", and no such shape moved
    a number when the two became fields. Qwen3 30B-A3B satisfies neither:
    it carries 32 heads of 128 at dim 2048, and an expert width of 768
    against a derived 7168. A registered shape that disagrees writes its
    own value; ``__post_init__`` fills the default, so after construction
    both are always an ``int``.

    ``PiperShape.derived`` applies the piper-1B family rules for a probe or a
    test shape. Never register a shape through it.
    """

    name: str
    dim: int
    n_layers: int
    head_dim: int
    n_kv_heads: int
    num_experts: int
    # Query heads. ``None`` derives ``dim // head_dim``, which every shape
    # registered before 30b-a3b satisfies and which ``__post_init__`` then
    # requires to be exact. It is not universal: Qwen3 30B-A3B carries 32
    # heads of 128 at dim 2048, so ``wo`` is ``[dim, n_heads * head_dim]``
    # and the two sides differ. Consumers must read this field and never
    # ``dim // head_dim``.
    n_heads: int | None = None
    # Expert width. ``None`` derives 3.5x dim (3584 at dim 1024, piper's
    # inter_dim), which piper 1B, 9B and 48B and the three synthetic shapes
    # all satisfy. Three real piper configs do not: 9M reads 128 against a
    # derived 896, and 30B-A3B reads 768 against a derived 7168, a 9.3x
    # error.
    moe_hidden_dim: int | None = None
    # Defaults, written by a shape that disagrees. Six registered shapes
    # take all four; 30b-a3b writes ``top_k`` 8. ``max_seq_len`` is the
    # harness's sequence ceiling and TorchTitan's RoPE cache size, not the
    # model's context length -- see PIPER_30B_A3B.
    top_k: int = 2
    vocab_size: int = 151936
    rope_theta: float = 1_000_000.0
    # Size of the precomputed CosSinRoPE cache, and the workload's sequence
    # ceiling. A longer sequence fails the eager bounds check; under compile
    # that check is dropped and the read goes out of bounds silently.
    max_seq_len: int = 4096
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
        # The two derived defaults. Each divisibility guard applies to the
        # derivation alone: a shape that writes its own value has no
        # quotient to keep exact. object.__setattr__ because the dataclass
        # is frozen; after this block both fields are always an int.
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
        """Build a probe shape from the piper-1B family rules.

        The rules are ``head_dim`` 64, ``n_kv_heads = n_heads // 2`` and 4
        experts, plus the two derived defaults ``n_heads = dim // head_dim``
        and ``moe_hidden_dim = dim * 7 // 2``, which this constructor never
        overrides. They describe piper 1B and no other real piper model: 9B
        and 48B carry 8 kv heads and 8 experts, 48B carries ``head_dim``
        128, and 30B-A3B breaks both derived defaults.

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
    def heads_per_group(self) -> int:
        return self.n_heads // self.n_kv_heads

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
        ratio with heads that tile dim. Piper 9B and 48B run 4:1, where the
        fused qkv is ``1.5*dim`` wide, so the closed form overcounts them;
        30B-A3B's heads do not tile dim, so ``wo`` is ``2*dim`` wide there.
        The two forms agree exactly on every 2:1 shape.
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
        """Parameters one rank holds, under the split both engines use.

        **The convention is that the embedding and the output head are not
        layers.** Megatron divides ``config.num_layers`` alone
        (``transformer_block.py``: ``account_for_embedding_in_pipeline_split``
        and its loss twin both default False), and the harness sends
        TorchTitan ``--parallelism.pipeline-parallel-first-stage-less-layers
        0`` and its ``last`` twin, which is that same arithmetic. So every
        stage holds ``n_layers // pipeline_degree`` layers, the first stage
        also holds the embedding table, and the last also holds the final
        norm and the untied output head.

        This is what makes validation arm rule 11 a real check under a
        pipeline split. Without it a rank could print any count: the whole
        model's, or its own, and nothing would separate a stage that built
        the wrong slice from one that built the right one. The sum over
        ``range(pipeline_degree)`` is ``param_count`` exactly, which is the
        other half a driver asserts.

        **``expert_degree`` is the second axis a rank is split along, and
        it is not the pipeline.** Under an expert degree a rank holds
        ``num_experts // expert_degree`` of the routed experts, so the
        expert term of every layer it owns divides. The router does NOT
        divide: it is the gate that chooses an expert, so every rank needs
        the whole of it. Nor does any dense parameter.

        **Without this term the guard refuses an honest expert-parallel
        run**, and it did: on 2026-08-28 a ``1b`` run at ``ep 2`` counted
        713,919,488 parameters against the 1,066,241,024 this function
        declared, and the driver raised. That number happens to equal
        ``nparams_active`` at ``1b`` because ``num_experts // expert_degree``
        is 2 there and ``top_k`` is 2 as well. **It is a coincidence of that
        shape and that degree, not a rule** -- at ``9b, ep 2`` a rank holds
        5,102,343,168 where ``nparams_active`` is 2,988,413,952. Never
        substitute one for the other.

        Rejects a degree the layer count does not divide, rather than
        rounding: an uneven split is a different model per rank, and
        ``validate_parallelism``'s rule 7 refuses such a run for the same
        reason. It rejects an expert degree the expert count does not
        divide for the same reason.
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
            # The final norm and the untied output head. The output head is a
            # second V x D table, which is why the two end stages are the
            # heavy ones at every shape this repo registers.
            count += self.dim + self.vocab_size * self.dim
        return count

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


# Piper 1B, verbatim. Every field matches examples/models/qwen3.py case
# '1B' in the piper checkout.
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

# Piper 9B, verbatim. Every field matches examples/models/qwen3.py case
# '9B' in the piper checkout.
#
# It is the first registered shape whose head geometry is not the 1B family's.
# 32 query heads over 8 kv heads is 4:1 grouped-query attention, where 1B and
# every synthetic shape run 2:1, so the fused qkv is 1.5x dim wide here rather
# than 2x. It is also the first with 8 experts.
#
# VALIDATION RULE 7 IS UNTESTED AT 24 LAYERS. The rule identifies a compiled
# block graph by its invocations per window, n_layers * profiler_active, which
# is 120 here against the 80 every published regioned run has. Nobody has
# matched 120 against a real trace. supports_block_regions is True, so a
# regioned scenario will declare the regions and rule 7 will judge them.
PIPER_9B = PiperShape(
    name="9b",
    dim=2048,
    n_layers=24,
    head_dim=64,
    n_kv_heads=8,
    num_experts=8,
    # UNVERIFIED, on the same sqrt(dim) law LARGE and GIANT use: dim 2048
    # predicts about 7.8e-3, and 2e-2 keeps huge's 2.46x margin over it. The
    # depth caveat on LARGE applies here and more so -- this is 24 layers,
    # further from both anchors than any other estimate in this file. Run
    # tools/megatron_parity_check.py --model-size 9b before any parity claim,
    # and --fp32-reference before you change this value.
    parity_gate=2e-2,
)

# Qwen3-30B-A3B, as examples/models/qwen3.py case '30B-A3B' in the piper
# checkout declares it (not the '-half' variant, which halves the layers).
# Every geometry field matches that case. Piper's registry declares this
# shape and its run archive holds no run of it.
#
# The shape that made n_heads and moe_hidden_dim fields. 32 heads of 128 at
# dim 2048 means n_heads * head_dim is 4096, twice dim: the query projection
# and wo are [2048, 4096], the fused qkv is (32 + 2*4) * 128 = 5120 wide,
# and dim // head_dim would have built 16 heads. The expert width is 768,
# where 3.5x dim would have built 7168, a 9.3x error in every expert tensor.
# It is also the first registered shape with 128 experts and top-8 routing,
# and the first with 8 query heads per kv head.
#
# WHAT THIS REGISTRATION DOES NOT CARRY. Piper declares max_seq_len 262144
# and a dense hidden_dim of 6144. The dense width is unused, because every
# layer is MoE, as it is at every other shape here. The context length is
# not carried: max_seq_len stays at the 4096 default, which is the harness's
# sequence ceiling AND the size of the CosSinRoPE cache config_registry.py
# builds from it. A run above seq 4096 needs that cache widened; nothing
# here widens it, and kernel-bench's --max-seq-len lifts the ceiling for the
# kernel side alone.
#
# NOTHING HAS RUN THIS SHAPE. No e2e scenario, no kernel scenario and no
# parity check has executed at it, in either engine. parity_gate is the
# default 2e-2 because no measurement exists to set it from; run
# tools/megatron_parity_check.py --model-size 30b-a3b before any parity
# claim. VALIDATION RULE 7 IS UNTESTED AT 48 LAYERS: the window invocation
# count is 240 here. THIS SHAPE DOES NOT FIT ONE H200: 30,532,122,624
# parameters at titan's 8 B/param of state is 227.5 GiB, and at megatron's
# 10 B/param under graph mode 284.4 GiB, against a 139.81 GiB device, before
# any activation. It divides evenly at pp 4 (12 layers a stage) and pp 8 (6
# a stage), and its 128 experts divide every expert degree up to 8.
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

# Piper 48B, verbatim. Every field matches examples/models/qwen3.py case
# '48B' in the piper checkout.
#
# The first registered shape whose head_dim is not 64. n_heads stays 32 and
# n_kv_heads stays 8, exactly as at 9B: piper widens the head and adds layers
# and experts, and holds the head counts.
#
# It shares dim 4096 and moe_hidden_dim 14336 with the synthetic LARGE, and
# differs in every remaining geometry value -- head_dim 128 against 64,
# n_heads 32 against 64, n_kv_heads 8 against 32, n_layers 32 against 4,
# num_experts 8 against 4. LARGE is 48B's width with the wrong head geometry
# and half the experts. Do not read one as an approximation of the other.
#
# VALIDATION RULE 7 IS UNTESTED AT 32 LAYERS, for the reason given on 9B: the
# window invocation count is 160 here.
#
# THIS SHAPE DOES NOT FIT ONE H200, AND IT IS NOT CLOSE. 47,685,316,608
# parameters at titan's 8 B/param of state (params 2 + grads 2 + fused-AdamW
# m,v 4) is 355 GiB, and at megatron's 10 B/param under graph mode it is 444
# GiB, against a 139.81 GiB device -- 2.5x to 3.2x over, before a single
# activation. GIANT is 2.8x smaller than this and already carries an OOM
# warning. Do not launch a sweep at 48b on one GPU; it is registered so the
# geometry is stated once and correctly, not because it is runnable here.
# 9B is the plausible one: 69.5 GiB titan, 86.9 GiB megatron, both unmeasured.
PIPER_48B = PiperShape(
    name="48b",
    dim=4096,
    n_layers=32,
    head_dim=128,
    n_kv_heads=8,
    num_experts=8,
    # UNVERIFIED, on the same law: dim 4096 predicts about 1.1e-2, and 3e-2
    # keeps huge's 2.46x margin. It is the same gate LARGE carries, which is
    # the arithmetic agreeing with itself -- the law reads dim alone and the
    # two shapes share a dim. Run tools/megatron_parity_check.py --model-size
    # 48b before any parity claim, and --fp32-reference before you change
    # this value.
    parity_gate=3e-2,
)

# Declared smallest to largest by parameter count. The names do not carry the
# order, so the declaration does.
PIPER_SHAPES: dict[str, PiperShape] = {
    shape.name: shape
    for shape in (
        PIPER_1B, LARGE, PIPER_9B, HUGE, GIANT, PIPER_30B_A3B, PIPER_48B
    )
}

# Retired ``--model-size`` names, each mapped to the key that replaced it.
#
# ``normal`` was the 1B shape's name until it took the real model's name. It
# stays accepted, and it must, because ``--resume`` compares the recorded
# string against the requested one. Measured under ``out/`` on 2026-08-21:
# 42 e2e manifests record ``"model_size": "normal"``, and 88 more carry no
# ``model_size`` at all and are defined to resume as that shape. A bare
# rename would refuse a resume that should succeed for 130 of the 144 e2e
# runs on disk. (13 kernel manifests also record ``"normal"``, but
# ``kernel-bench`` has no resume, so they are not part of this argument.)
# Re-derive the counts rather than quoting them:
#   grep -l '"model_size": "normal"' $(find out -name manifest.json)
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
