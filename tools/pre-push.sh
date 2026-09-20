#!/usr/bin/env bash
# Pre-push hook: run the CPU test suite before a push leaves this machine.
#
# Git runs a hook from the top of the pushing worktree, so
# `git rev-parse --show-toplevel` names that worktree. One hook file is
# shared by every worktree of this repository, and each one tests itself.
#
# CUDA_VISIBLE_DEVICES is empty on purpose: the GPU tests skip themselves,
# the suite takes about a minute, and a push must not touch a GPU another
# run is using.
#
# Bypass with `git push --no-verify`, or with PRE_PUSH_SKIP=1.
set -euo pipefail

if [ "${PRE_PUSH_SKIP:-0}" = "1" ]; then
    echo "pre-push: PRE_PUSH_SKIP=1, the test suite is skipped." >&2
    exit 0
fi

repo_root="$(git rev-parse --show-toplevel)"
python="$repo_root/.venv/bin/python"

if [ ! -x "$python" ]; then
    echo "pre-push: no environment at $python; the test suite is skipped." >&2
    exit 0
fi

cd "$repo_root"
export CUDA_VISIBLE_DEVICES=""
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/hf-datasets}"
mkdir -p "$HF_DATASETS_CACHE"

if ! "$python" -m unittest discover -s tests; then
    echo "pre-push: the test suite failed, so the push is refused; use 'git push --no-verify' to override." >&2
    exit 1
fi
