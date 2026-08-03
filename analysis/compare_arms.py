"""Compatibility entry point for the integrated benchmark evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks import metrics as _metrics  # noqa: E402
from benchmarks.cli import legacy_evaluate_main  # noqa: E402


# Compatibility exports for code that imported helpers from this script.
describe = _metrics.describe
evaluate_run = _metrics.evaluate_run
grad_norms = _metrics.grad_norms
losses = _metrics.losses
region_comparison = _metrics.region_comparison
stable_tps = _metrics.stable_tps
training_metrics = _metrics.training_metrics


def load_run(out_dir, arms_override):
    """Preserve the former three-value analyzer helper API."""
    manifest, arms, regions, warnings = _metrics.load_run(out_dir, arms_override)
    for warning in warnings:
        print(f"WARNING: {warning}")
    return manifest, arms, regions


def main() -> None:
    legacy_evaluate_main()


if __name__ == "__main__":
    main()
