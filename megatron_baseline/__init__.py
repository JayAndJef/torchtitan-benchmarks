"""Megatron-LM baseline arm for the piper-1B benchmarks.

Everything Megatron-specific lives in this package: locating the Megatron-LM
checkout (location), building the Qwen3-1B model as a bare megatron-core
GPTModel (model), materializing the exact TorchTitan c4_test batch stream in
THD packed form (data), and the training driver the benchmark runner launches
(train). Nothing outside this package imports megatron; the harness connects
only through the arm's launch command and megatron_baseline.location for
provenance.

Importing this package (and every module except at driver runtime) does not
require megatron or Transformer Engine to be installed: all megatron imports
happen inside functions.
"""
