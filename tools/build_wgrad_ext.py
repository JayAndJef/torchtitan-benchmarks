"""Build apex's ``fused_weight_gradient_mlp_cuda`` alone, for stock Megatron.

Stock Megatron enables gradient accumulation fusion by default. Its
``ColumnParallelLinear`` output layer then imports this one apex extension.
Apex offers no switch for one core extension: ``APEX_CUDA_EXT=1`` builds all
eleven, and one failure stops the whole install. So this script compiles the
three vendored sources under ``third_party/apex-wgrad/`` and nothing else.

    source /opt/rh/gcc-toolset-13/enable
    .venv/bin/python tools/build_wgrad_ext.py [--check]

The module and a stamp go to ``.apex-wgrad/``, which is gitignored and not
part of the venv, so ``uv sync`` never removes it. The stamp records the
torch version and the source digest. The stock driver refuses a missing or
stale build, so a torch pin bump needs this script again. ``sync.sh`` runs it.

``--check`` also compares both kernels against a plain PyTorch GEMM on a GPU,
and fails when no GPU is visible. Without it, the script checks the import.

The new build replaces the old one only after the checks pass.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.e2e.megatron_stock.bootstrap import (  # noqa: E402
    WGRAD_BUILD_DIR,
    WGRAD_MODULE,
    WGRAD_SOURCE_DIR,
    WGRAD_SOURCES,
    wgrad_expected_stamp,
)

CUDA_ARCH = "9.0"
"""Every GPU here is sm_90, so the build targets it alone."""

NVCC_FLAGS = (
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
)
"""The nvcc flags apex's ``setup.py`` gives this extension."""


def fail(message: str) -> None:
    """Print one error line and exit nonzero."""
    print(f"build_wgrad_ext: error: {message}", file=sys.stderr)
    raise SystemExit(1)


def prepare_environment() -> None:
    """Point the build at the venv's CUDA toolkit, ninja and C++ compiler."""
    cuda_home = Path(os.environ.get("CUDA_HOME", REPO_ROOT / ".cuda-home"))
    if not (cuda_home / "bin" / "nvcc").exists():
        fail(
            f"no nvcc under CUDA_HOME={cuda_home}; run ./sync.sh once, which "
            "assembles .cuda-home/ from the venv's CUDA wheels"
        )
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH
    venv_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = f"{cuda_home / 'bin'}:{venv_bin}:{os.environ['PATH']}"
    toolset = Path("/opt/rh/gcc-toolset-13/root/usr/bin/g++")
    if "CXX" not in os.environ and toolset.exists():
        os.environ["CXX"] = str(toolset)


def compile_module(work_dir: Path):
    """Compile the extension into ``work_dir`` and return the loaded module."""
    from torch.utils.cpp_extension import load

    return load(
        name=WGRAD_MODULE,
        sources=[str(WGRAD_SOURCE_DIR / source) for source in WGRAD_SOURCES],
        extra_include_paths=[str(WGRAD_SOURCE_DIR / "csrc")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=list(NVCC_FLAGS),
        build_directory=str(work_dir),
        verbose=False,
    )


def check_kernels(module) -> None:
    """Compare both kernels against ``d_weight += d_output.T @ input``."""
    import torch

    if not torch.cuda.is_available():
        fail("--check needs a visible GPU")
    torch.manual_seed(0)
    tokens, fan_in, fan_out = 512, 256, 384
    for name, grad_dtype in (
        ("wgrad_gemm_accum_fp32", torch.float32),
        ("wgrad_gemm_accum_fp16", torch.bfloat16),
    ):
        inputs = torch.randn(tokens, fan_in, device="cuda", dtype=torch.bfloat16)
        d_output = torch.randn(tokens, fan_out, device="cuda", dtype=torch.bfloat16)
        d_weight = torch.randn(fan_out, fan_in, device="cuda", dtype=grad_dtype)
        expected = d_weight.float() + d_output.float().t() @ inputs.float()
        getattr(module, name)(inputs, d_output, d_weight)
        torch.cuda.synchronize()
        error = (d_weight.float() - expected).norm() / expected.norm()
        limit = 1e-3 if grad_dtype == torch.float32 else 1e-2
        if not error <= limit:
            fail(f"{name}: relative error {error:.2e} exceeds {limit:.0e}")
        print(f"build_wgrad_ext: {name} matches torch (rel_l2 {error:.2e})")


def install(work_dir: Path) -> Path:
    """Move the built library and a new stamp into ``WGRAD_BUILD_DIR``."""
    built = sorted(work_dir.glob(f"{WGRAD_MODULE}*.so"))
    if len(built) != 1:
        fail(f"expected one {WGRAD_MODULE} library in {work_dir}, found {built}")
    staging = WGRAD_BUILD_DIR.with_name(WGRAD_BUILD_DIR.name + ".new")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    shutil.copy2(built[0], staging / built[0].name)
    (staging / "stamp.json").write_text(
        json.dumps(wgrad_expected_stamp(), indent=2) + "\n"
    )
    shutil.rmtree(WGRAD_BUILD_DIR, ignore_errors=True)
    staging.rename(WGRAD_BUILD_DIR)
    return WGRAD_BUILD_DIR / built[0].name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="also compare both kernels against torch on a GPU",
    )
    args = parser.parse_args()
    prepare_environment()
    with tempfile.TemporaryDirectory(prefix="apex-wgrad-") as scratch:
        module = compile_module(Path(scratch))
        if args.check:
            check_kernels(module)
        library = install(Path(scratch))
    print(f"build_wgrad_ext: built {library}")


if __name__ == "__main__":
    main()
