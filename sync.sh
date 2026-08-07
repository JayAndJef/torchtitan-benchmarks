#!/usr/bin/env bash
# uv sync wrapper. transformer-engine-torch (megatron group) builds without
# isolation against the pinned torch nightly and needs the venv's bundled
# NVIDIA headers (nccl/cudnn/cu13) plus a C++20 host compiler. The first
# pass installs torch and those header wheels; the second builds TE.
set -euo pipefail
cd "$(dirname "$0")"
if [ -f /opt/rh/gcc-toolset-13/enable ]; then
    source /opt/rh/gcc-toolset-13/enable
fi
uv sync --no-group megatron "$@"
NV="$(echo "$PWD"/.venv/lib/python*/site-packages/nvidia)"
export CPATH="$NV/nccl/include:$NV/cudnn/include:$NV/cu13/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$NV/nccl/lib:$NV/cudnn/lib:$NV/cu13/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
uv sync "$@"
