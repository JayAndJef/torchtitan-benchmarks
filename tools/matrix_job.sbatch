#!/usr/bin/env bash
# One Slurm job that runs the cells of a benchmark matrix, one after another.
#
# This file is a template, and a working example: the dp4 x ep4 matrix of
# the stacked scenario, with seven cells. Cells a and b repeat one arm.
#
# Copy the template into a new output root, and edit the copy:
#   mkdir -p out/<root>
#   cp tools/matrix_job.sbatch out/<root>/matrix.sbatch
# Set ROOT, the #SBATCH lines, the device list, the shared flags and the
# cell lines. Read the admin skill run-gpu-job for the partition and the
# time limit. The `dp * pp` of each cell must equal the number of cards.
#
# Submit the copy from the repository root, on a clean tree:
#   MATRIX_REV=$(git rev-parse HEAD) sbatch -o out/<root>/slurm-%j.out out/<root>/matrix.sbatch
#
# tools/run_matrix_cell.sh runs each cell, and its header lists the states.
# Each cell writes one line `STATUS <name> <state>` to out/<root>/sweep.log.
# A failed cell does not stop the job. To retry the cells that are not OK,
# submit the same copy again; a cell with a results.json is skipped. Retry
# at most 3 times. Never report a CONTAMINATED cell.
#SBATCH -p main
#SBATCH --gres=gpu:idx4:1,gpu:idx5:1,gpu:idx6:1,gpu:idx7:1
#SBATCH --cpus-per-task=128
#SBATCH --mem=400000M
#SBATCH --time=01:15:00
#SBATCH --no-requeue
#SBATCH -J matrix
set -uo pipefail

fail() { echo "matrix_job.sbatch: $*" >&2; exit 2; }

ROOT=out/CHANGE-ME
[ "$ROOT" != out/CHANGE-ME ] || fail "set ROOT in the copy of this file"

# Slurm runs a copy of this file from its spool directory, so the job finds
# the repository through the directory of the submission.
cd "${SLURM_SUBMIT_DIR:?this file runs only as a Slurm job}" || exit 2
[ -x tools/run_matrix_cell.sh ] || fail "submit from the repository root"
[ -d "$ROOT" ] || fail "the output root $ROOT is not a directory"

# Each cell gets its own --out. An exported WARMUP_STEPS refuses a profiled run.
unset OUT WARMUP_STEPS

DEVICES=0,1,2,3
C="--model-size 30b-a3b-20l --dp 4 --ep 4 --zero 1 --batch 4 --profile --steps 80"
F="--set megatron_stock.extra_flags+=--cross-entropy-loss-fusion --set megatron_stock.extra_flags+=--moe-permute-fusion --set megatron_stock.extra_flags+=--overlap-grad-reduce --set megatron_stock.extra_flags+=--overlap-param-gather"

# cell <name> <run flags>. A failed cell does not stop the job, and the cell
# runner writes the STATUS line.
cell() {
    local name="$1"
    shift
    tools/run_matrix_cell.sh "$ROOT" "$name" run "$DEVICES" "$@" || true
}

# The flag strings are unquoted on purpose: the word split turns each one
# into arguments. So a value may hold no space and no quote.
cell 1-titan_compiled-a                 --scenario stacked   --arm titan_compiled $C
cell 2-titan_compiled_fa3               --scenario attention --arm titan_compiled_fa3 $C
cell 3-titan_compiled_te_per_expert     --scenario experts   --arm titan_compiled_te_per_expert $C
cell 4-titan_compiled_fa3_te_per_expert --scenario stacked   --arm titan_compiled_fa3_te_per_expert $C
cell 5-megatron_stock                   --scenario stacked   --arm megatron_stock $C
cell 6-megatron_stock_4flags            --scenario stacked   --arm megatron_stock $F $C
cell 7-titan_compiled-b                 --scenario stacked   --arm titan_compiled $C
