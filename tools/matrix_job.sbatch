#!/usr/bin/env bash
# One Slurm job that runs the cells of a benchmark matrix, one after another.
#
# This file is a template, and a working example: a dp4 x ep4 matrix of the
# engines and attention arms, with six cells. Cells a and b repeat one arm,
# so their spread shows the run-to-run noise.
#
# Copy the template into a new output root, and edit the copy:
#   mkdir -p out/<root>
#   cp tools/matrix_job.sbatch out/<root>/matrix.sbatch
# Set ROOT, the #SBATCH lines, the device list, the shared flags and the
# cell lines. Read the admin skill run-gpu-job for the partition and the
# time limit. The `dp * pp` of each cell must equal the number of cards.
# Each cell needs exactly one --scenario, and a name of [A-Za-z0-9_-]+.
#
# The --time must cover the sum of the run times of the cells, plus up to
# IDLE_MAX_WAIT seconds of load gate per cell. With the default of 1200 s,
# the gate alone can take 2 h for six cells, so this example sets 300 s.
# The 2026-10-07 job ran seven such cells in about 26 min.
#
# Submit the copy from the repository root, on a clean tree:
#   MATRIX_REV=$(git rev-parse HEAD) sbatch -o out/<root>/slurm-%j.out out/<root>/matrix.sbatch
#
# tools/run_matrix_cell.sh runs each cell, and its header lists the states.
# Each cell writes one line `STATUS <name> <state>` to out/<root>/sweep.log.
# A failed cell does not stop the job. To retry the cells that are not OK,
# submit the same copy again; a done cell is skipped. Retry only a
# coincidental failure, at most 3 times. Never report a CONTAMINATED cell.
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
: "${MATRIX_REV:?submit with MATRIX_REV=$(git rev-parse HEAD)}"
[ -x tools/run_matrix_cell.sh ] || fail "submit from the repository root"
[ -d "$ROOT" ] || fail "the output root $ROOT is not a directory"

# Each cell gets its own --out. An exported WARMUP_STEPS refuses a profiled run.
unset OUT WARMUP_STEPS

# Six cells at about 4 min each, plus at most 5 min of load gate each, fit
# the --time above.
export IDLE_MAX_WAIT=300

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
cell 1-titan_compiled-a        --scenario engines   --arm titan_compiled $C
cell 2-titan_eager             --scenario engines   --arm titan_eager $C
cell 3-titan_compiled_fa3      --scenario attention --arm titan_compiled_fa3 $C
cell 4-megatron_stock          --scenario engines   --arm megatron_stock $C
cell 5-megatron_stock_4flags   --scenario engines   --arm megatron_stock $F $C
cell 6-titan_compiled-b        --scenario engines   --arm titan_compiled $C
