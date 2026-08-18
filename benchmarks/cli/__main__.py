"""``python -m benchmarks.cli`` entry point; run_bench.sh execs this."""

from benchmarks.cli.main import cli

if __name__ == "__main__":
    cli(prog_name="run_bench.sh")
