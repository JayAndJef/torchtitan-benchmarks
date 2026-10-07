#!/usr/bin/env bash
# uv sync wrapper. Two dependency groups build CUDA code without isolation
# against the pinned torch and need the venv's bundled NVIDIA headers
# plus a C++20 host compiler:
#
#   megatron  transformer-engine-torch (prebuilt core wheel, torch binding
#             compiles at install)
#   flash3    flash-attn-3, a full source build of CUTLASS sm90a kernels from
#             the flash-attention repo's hopper/ subdirectory (15-40 min).
#
# uv caches both builds by their source alone and ignores the torch they built
# against. The first pass also uninstalls both, so the second pass reinstalls
# them from the cache. So each torch pin gets its own uv cache, and a pin bump
# rebuilds both against the new torch. Delete the old pin's cache, which holds
# several GiB.
#
# The first pass installs torch and the header wheels; the second builds both.
# Skip the long one with:  ./sync.sh --no-group flash3
#
# The third group, fa4 (flash-attn-4 + the CuTe DSL), needs none of this: FA4
# generates its kernels at compile time and ships pure-Python wheels.
set -euo pipefail
cd "$(dirname "$0")"
torch_pin="$(grep -oE '"torch==[^" ;]+' pyproject.toml | tr -d '"' || true)"
if [ "$(printf '%s\n' "$torch_pin" | grep -c .)" -ne 1 ]; then
    echo "sync.sh: expected one torch== pin in pyproject.toml, found: ${torch_pin:-none}" >&2
    exit 1
fi
uv_cache_base="${UV_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/uv}"
export UV_CACHE_DIR="${uv_cache_base%/}-${torch_pin#torch==}"
echo "sync.sh: uv cache $UV_CACHE_DIR"
if [ -f /opt/rh/gcc-toolset-13/enable ]; then
    source /opt/rh/gcc-toolset-13/enable
fi
uv sync --no-group megatron --no-group flash3 "$@"
NV="$(echo "$PWD"/.venv/lib/python*/site-packages/nvidia)"
export CPATH="$NV/nccl/include:$NV/cudnn/include:$NV/cu13/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$NV/nccl/lib:$NV/cudnn/lib:$NV/cu13/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

# FA3 additionally needs a CUDA_HOME that looks like a real toolkit. The pip
# nvcc is a subset (bin/nvcc only) and the cu13 wheel ships versioned sonames
# without the unversioned linker names `-lcudart` resolves against, so
# assemble both as a symlink tree under .cuda-home/.
SHIM="$PWD/.cuda-home"
rm -rf "$SHIM"
mkdir -p "$SHIM/lib64"
ln -s "$NV/cu13/bin" "$SHIM/bin"
ln -s "$NV/cu13/include" "$SHIM/include"
ln -s "$NV/cu13/nvvm" "$SHIM/nvvm"
for lib in "$NV/cu13/lib"/*; do
    ln -sf "$lib" "$SHIM/lib64/$(basename "$lib")"
done
for versioned in "$NV/cu13/lib"/lib*.so.*; do
    linkname="$SHIM/lib64/$(basename "${versioned%%.so.*}").so"
    [ -e "$linkname" ] || ln -s "$versioned" "$linkname"
done
export CUDA_HOME="$SHIM"
# .venv/bin carries ninja; without it cpp_extension falls back to the
# single-threaded distutils backend and the FA3 build takes hours.
export PATH="$SHIM/bin:$PWD/.venv/bin:$PATH"
export LIBRARY_PATH="$SHIM/lib64:$LIBRARY_PATH"
# Every GPU here is sm_90; sm_80 objects can never run and roughly double the
# compile. The dtype/head-dim matrix stays complete.
export FLASH_ATTENTION_DISABLE_SM80="${FLASH_ATTENTION_DISABLE_SM80:-TRUE}"
export MAX_JOBS="${MAX_JOBS:-64}"
export NVCC_THREADS="${NVCC_THREADS:-2}"

uv sync "$@"

# Stock Megatron's gradient accumulation fusion needs one apex extension.
# Apex builds its eleven core extensions all or none, so this builds the one
# from vendored sources into .apex-wgrad/, outside the venv.
.venv/bin/python tools/build_wgrad_ext.py

# Install the pre-push hook. Worktrees share one hook set, so the common dir
# is the correct target; --git-dir would name the per-worktree directory and
# the hook would apply to this worktree alone. The link is absolute and the
# command is idempotent.
repo_root="$PWD"
hooks_dir="$(cd "$(git rev-parse --git-common-dir)" && pwd)/hooks"
mkdir -p "$hooks_dir"
ln -sfn "$repo_root/tools/pre-push.sh" "$hooks_dir/pre-push"
echo "sync.sh: linked $hooks_dir/pre-push to $repo_root/tools/pre-push.sh"
