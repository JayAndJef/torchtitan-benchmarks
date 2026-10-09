# Torch's bundled cuDNN for every process, TransformerEngine included.
#
# TransformerEngine links the cuDNN parts by soname and opens libcudnn.so by
# name. Both resolve to the system cuDNN in /usr/lib64 when nothing else
# comes first, so a process maps the system cuDNN beside torch's bundled
# one, and torch.backends.cudnn.version() raises. This script sets
# CUDNN_HOME, which TransformerEngine searches first, and puts its lib
# directory first on LD_LIBRARY_PATH.
#
# Source it (run_bench.sh does this automatically):
#   source ./cudnn_env.sh

_cudnn_python="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv/bin/python"
# Ask the interpreter, so the path follows the venv's Python version.
CUDNN_HOME="$("$_cudnn_python" -I -c 'import nvidia.cudnn; print(nvidia.cudnn.__path__[0])')" &&
    [ -e "$CUDNN_HOME/lib/libcudnn.so.9" ] ||
    {
        echo "cudnn_env: no bundled cuDNN in $_cudnn_python's nvidia.cudnn package; run ./sync.sh" >&2
        return 1 2>/dev/null || exit 1
    }
export CUDNN_HOME
case ":${LD_LIBRARY_PATH:-}:" in
":$CUDNN_HOME/lib:"*) ;;
*) export LD_LIBRARY_PATH="$CUDNN_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
