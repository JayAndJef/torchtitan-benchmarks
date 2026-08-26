"""The stock Megatron-LM arm: the driver and the pieces it needs.

The tuned arm in ``benchmarks/e2e/megatron/`` replicates the TorchTitan
treatment step by step. This package does the opposite: it hands the run to
Megatron's own ``megatron.training.pretrain`` and to ``pretrain_gpt``'s own
providers, and substitutes one argument only -- the dataset provider, so
that both engines read the same c4_test stream rank for rank.

Importing this package pulls in no torch and no megatron. Every megatron
import happens inside a function, and ``flags.py`` imports neither.
"""
