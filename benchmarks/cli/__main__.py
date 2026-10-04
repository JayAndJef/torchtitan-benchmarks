"""The ``python -m benchmarks.cli`` entry point, which ``run_bench.sh`` starts."""

from benchmarks.cli.main import cli

if __name__ == "__main__":
    cli(prog_name="run_bench.sh")
