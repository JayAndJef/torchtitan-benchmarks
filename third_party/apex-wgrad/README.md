# apex-wgrad

Four files of NVIDIA apex, vendored so that `tools/build_wgrad_ext.py` can
build `fused_weight_gradient_mlp_cuda` alone. Stock Megatron needs that
module for gradient accumulation fusion.

- Upstream: https://github.com/NVIDIA/apex
- Commit: `8a6508aaad6e75a2b939e33f308cd63d745d97f1` (2026-09-14)
- Files: `csrc/type_shim.h` and the three `csrc/megatron/fused_weight_gradient_dense*`
  sources, byte for byte. `LICENSE` is apex's BSD-3-Clause license.

Do not edit these files. To update them, copy the same paths from a newer
apex commit, change the commit above, and run the build script again.
