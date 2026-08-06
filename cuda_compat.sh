# CUDA forward-compat userspace driver for the pinned cu130 wheels.
#
# The torch pin is cu130, but this box's kernel driver (570.211.01) reports
# CUDA 12.8 in the nvidia-smi header. NVIDIA's forward-compat package layers
# a newer userspace libcuda over the older kernel module. This script stages
# those libraries under .cuda-compat/ (gitignored) and prepends them to
# LD_LIBRARY_PATH -- but only when the driver actually needs it.
#
# Source it (run_bench.sh does this automatically):
#   source ./cuda_compat.sh
# Or run it to stage the libraries and print the export line:
#   bash ./cuda_compat.sh

_compat_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.cuda-compat"
_compat_rpm="cuda-compat-13-0-580.178.04-1.el9.x86_64.rpm"
_compat_url="https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/${_compat_rpm}"

_cuda_compat_needed() {
    # cu130 needs the r580+ userspace driver. Check the kernel driver
    # version, not the nvidia-smi CUDA header: the header flips to 13.0 as
    # soon as the compat libraries themselves are on LD_LIBRARY_PATH.
    local driver major
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
        2>/dev/null | head -n1)" || return 0
    major="${driver%%.*}"
    case "$major" in "" | *[!0-9]*) return 0 ;; esac
    [ "$major" -ge 580 ] && return 1
    return 0
}

_cuda_compat_stage() {
    [ -e "$_compat_dir/libcuda.so.1" ] && return 0
    echo "cuda_compat: staging $_compat_rpm into $_compat_dir" >&2
    local tmp
    tmp="$(mktemp -d)" &&
        curl -fsSL "$_compat_url" -o "$tmp/$_compat_rpm" &&
        (cd "$tmp" && rpm2cpio "$_compat_rpm" | cpio -idm --quiet) &&
        mkdir -p "$_compat_dir" &&
        cp -a "$tmp"/usr/local/cuda-13.0/compat/. "$_compat_dir/" &&
        rm -rf "$tmp" &&
        [ -e "$_compat_dir/libcuda.so.1" ] ||
        {
            echo "cuda_compat: failed to stage compat libraries" >&2
            return 1
        }
}

if _cuda_compat_needed; then
    _cuda_compat_stage || return 1 2>/dev/null || exit 1
    case ":${LD_LIBRARY_PATH:-}:" in
    *":$_compat_dir:"*) ;;
    *) export LD_LIBRARY_PATH="$_compat_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi

# Executed rather than sourced: print what a shell should export.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    if _cuda_compat_needed; then
        echo "export LD_LIBRARY_PATH=$_compat_dir\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
    else
        echo "# driver already reports CUDA 13.0+; no compat needed"
    fi
fi
