"""Megatron-LM baseline arm: THD data and the training driver.

This package holds the two Megatron-specific pieces the benchmark runner
drives directly: materializing the exact TorchTitan c4_test batch stream in
THD packed form (data), and the training driver the runner launches (train).
Locating the Megatron-LM checkout (megatron_bootstrap) and building the
Qwen3-1B model as a bare megatron-core GPTModel (megatron_model) live beside
the shape they are built from, in benchmarks/models/piper_qwen3/. The harness
connects only through the arm's launch command and
benchmarks.models.piper_qwen3.megatron_bootstrap for provenance.

Importing this package (and every module except at driver runtime) does not
require megatron or Transformer Engine to be installed: all megatron imports
happen inside functions.
"""
