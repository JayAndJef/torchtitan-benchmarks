// Standalone port of NVIDIA TransformerEngine's fused RoPE kernels.
//
// Device code below is copied VERBATIM from
// TransformerEngine/transformer_engine/common/fused_rope/fused_rope.cu
// (fused_rope_block_forward, fused_rope_block_backward,
//  fused_rope_forward_kernel, fused_rope_backward_kernel).
// Only the surrounding plumbing is new: TE-internal headers (common.h,
// utils.cuh, transformer_engine/fused_rope.h) are replaced by a local
// THREADS_PER_WARP constant and a thin torch::Tensor wrapper that reproduces
// TE's fused_rope_{forward,backward}_launcher launch configuration for the
// BSHD tensor format.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int THREADS_PER_WARP = 32;  // from TE's transformer_engine/common/utils.cuh

// ---------------------------------------------------------------------------
// BEGIN verbatim TE device code
// ---------------------------------------------------------------------------

template <typename scalar_t>
__device__ void fused_rope_block_forward(const scalar_t *src, const float *freqs, scalar_t *dst,
                                         const bool interleaved, const int s_id,
                                         const int offset_block, const int offset_block_dst,
                                         const int h, const int d, const int d2, const int stride_h,
                                         const int stride_d, const int o_stride_h,
                                         const int o_stride_d) {
  extern __shared__ float shared_mem_cos_sin[];
  float *shared_mem_cos = shared_mem_cos_sin;
  float *shared_mem_sin = shared_mem_cos_sin + d2;
  int tid = threadIdx.x * blockDim.y + threadIdx.y;
  for (int i = tid; i < d2; i += blockDim.x * blockDim.y) {
    sincosf(freqs[s_id * d2 + i], &shared_mem_sin[i], &shared_mem_cos[i]);
  }
  __syncthreads();

#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
      float v_cos = shared_mem_cos[d_id];
      float v_sin = shared_mem_sin[d_id];
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate;
      if (!interleaved) {
        v_src_rotate = (d_id + d2 / 2 < d2)
                           ? -static_cast<float>(src[offset_src + (d2 / 2) * stride_d])
                           : static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
      } else {
        v_src_rotate = (d_id % 2 == 0)
                           // d_id + 1
                           ? -static_cast<float>(src[offset_src + stride_d])
                           // d_id - 1
                           : static_cast<float>(src[offset_src - stride_d]);
      }
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
#pragma unroll
      for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
        int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
        int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
        dst[offset_dst] = src[offset_src];
      }
    }
  }
}

template <typename scalar_t>
__device__ void fused_rope_block_backward(const scalar_t *src, const float *freqs, scalar_t *dst,
                                          const bool interleaved, const int s_id,
                                          const int offset_block, const int offset_block_dst,
                                          const int h, const int d, const int d2,
                                          const int stride_h, const int stride_d,
                                          const int o_stride_h, const int o_stride_d) {
  extern __shared__ float shared_mem_cos_sin[];
  float *shared_mem_cos = shared_mem_cos_sin;
  float *shared_mem_sin = shared_mem_cos_sin + d2;
  int tid = threadIdx.x * blockDim.y + threadIdx.y;
  for (int i = tid; i < d2; i += blockDim.x * blockDim.y) {
    sincosf(freqs[s_id * d2 + i], &shared_mem_sin[i], &shared_mem_cos[i]);
  }
  __syncthreads();

#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_cos = shared_mem_cos[d_id];
      float v_src_rotate, v_sin;
      if (!interleaved) {
        if (d_id + d2 / 2 < d2) {
          v_src_rotate = static_cast<float>(src[offset_src + (d2 / 2) * stride_d]);
          v_sin = shared_mem_sin[d_id + d2 / 2];
        } else {
          v_src_rotate = static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
          v_sin = -shared_mem_sin[d_id + d2 / 2 - d2];
        }
      } else {
        if (d_id % 2 == 0) {
          v_src_rotate = static_cast<float>(src[offset_src + stride_d]);
          v_sin = shared_mem_sin[d_id + 1];
        } else {
          v_src_rotate = static_cast<float>(src[offset_src - stride_d]);
          v_sin = -shared_mem_sin[d_id - 1];
        }
      }
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
#pragma unroll
      for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
        int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
        int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
        dst[offset_dst] = src[offset_src];
      }
    }
  }
}

template <typename scalar_t>
__global__ void fused_rope_forward_kernel(const scalar_t *src, const int *cu_seqlens,
                                          const float *freqs, const int *start_positions,
                                          scalar_t *dst, const bool interleaved, const int cp_size,
                                          const int cp_rank, const int s, const int h, const int d,
                                          const int d2, const int stride_s_or_t, const int stride_b,
                                          const int stride_h, const int stride_d,
                                          const int o_stride_s_or_t, const int o_stride_b,
                                          const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block, offset_block_dst;
  int cur_seqlens;
  if (cu_seqlens != nullptr) {  // THD
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    offset_block = t_id * stride_s_or_t;
    offset_block_dst = t_id * o_stride_s_or_t;
    cur_seqlens = end - start;
  } else {  // SBHD/BSHD
    offset_block = s_id * stride_s_or_t + b_id * stride_b;
    offset_block_dst = s_id * o_stride_s_or_t + b_id * o_stride_b;
    cur_seqlens = s;
  }

  // Offset the RoPE embedding by start_positions if provided.
  int begin_offset = (start_positions == nullptr) ? 0 : start_positions[b_id];
  int s_id_for_freqs = s_id + begin_offset;

  // If CP_SIZE > 1, offset the RoPE embedding by cp_rank based on the dual-chunk order.
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs += cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs += cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
    }
  }

  fused_rope_block_forward(src, freqs, dst, interleaved, s_id_for_freqs, offset_block,
                           offset_block_dst, h, d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
__global__ void fused_rope_backward_kernel(
    const scalar_t *src, const int *cu_seqlens, const float *freqs, const int *start_positions,
    scalar_t *dst, const bool interleaved, const int cp_size, const int cp_rank, const int s,
    const int h, const int d, const int d2, const int stride_s_or_t, const int stride_b,
    const int stride_h, const int stride_d, const int o_stride_s_or_t, const int o_stride_b,
    const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block, offset_block_dst;
  int cur_seqlens;
  if (cu_seqlens != nullptr) {  // THD
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    offset_block = t_id * stride_s_or_t;
    offset_block_dst = t_id * o_stride_s_or_t;
    cur_seqlens = end - start;
  } else {  // SBHD/BSHD
    offset_block = s_id * stride_s_or_t + b_id * stride_b;
    offset_block_dst = s_id * o_stride_s_or_t + b_id * o_stride_b;
    cur_seqlens = s;
  }

  // Offset the RoPE embedding by start_positions if provided.
  int begin_offset = (start_positions == nullptr) ? 0 : start_positions[b_id];
  int s_id_for_freqs = s_id + begin_offset;

  // If CP_SIZE > 1, offset the RoPE embedding by cp_rank based on the dual-chunk order.
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs += cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs += cur_seqlens * cp_size - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
    }
  }

  fused_rope_block_backward(src, freqs, dst, interleaved, s_id_for_freqs, offset_block,
                            offset_block_dst, h, d, d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

// ---------------------------------------------------------------------------
// END verbatim TE device code
// ---------------------------------------------------------------------------

// Piper specialization: select the frequency row from a per-token position
// tensor. The TE block rotation remains unchanged; only sequence indexing is
// specialized for contiguous BSHD tensors without context parallelism.
template <typename scalar_t>
__global__ void fused_rope_forward_positions_kernel(
    const scalar_t *src, const int64_t *positions, const float *freqs, scalar_t *dst,
    const bool interleaved, const int s, const int h, const int d, const int d2,
    const int stride_s, const int stride_b, const int stride_h, const int stride_d,
    const int o_stride_s, const int o_stride_b, const int o_stride_h, const int o_stride_d) {
  const int s_id = blockIdx.x;
  const int b_id = blockIdx.y;
  const int position = static_cast<int>(positions[b_id * s + s_id]);
  const int offset_block = s_id * stride_s + b_id * stride_b;
  const int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  fused_rope_block_forward(src, freqs, dst, interleaved, position, offset_block,
                           offset_block_dst, h, d, d2, stride_h, stride_d, o_stride_h,
                           o_stride_d);
}

template <typename scalar_t>
__global__ void fused_rope_backward_positions_kernel(
    const scalar_t *src, const int64_t *positions, const float *freqs, scalar_t *dst,
    const bool interleaved, const int s, const int h, const int d, const int d2,
    const int stride_s, const int stride_b, const int stride_h, const int stride_d,
    const int o_stride_s, const int o_stride_b, const int o_stride_h, const int o_stride_d) {
  const int s_id = blockIdx.x;
  const int b_id = blockIdx.y;
  const int position = static_cast<int>(positions[b_id * s + s_id]);
  const int offset_block = s_id * stride_s + b_id * stride_b;
  const int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  fused_rope_block_backward(src, freqs, dst, interleaved, position, offset_block,
                            offset_block_dst, h, d, d2, stride_h, stride_d, o_stride_h,
                            o_stride_d);
}

// DIAGNOSTIC ABLATION ONLY -- produces WRONG results by design.
// Identical to fused_rope_block_forward / fused_rope_forward_kernel except that
// the rotate-half partner load is dropped (v_src is reused in its place). This
// isolates the cost of TE's second scalar load per element from everything else
// (grid shape, scalar access width, shared-memory sincos). It is never used for
// any correctness or headline performance claim.
template <typename scalar_t>
__device__ void ablate_block_forward_no_partner_load(const scalar_t *src, const float *freqs,
                                                     scalar_t *dst, const int s_id,
                                                     const int offset_block,
                                                     const int offset_block_dst, const int h,
                                                     const int d, const int d2, const int stride_h,
                                                     const int stride_d, const int o_stride_h,
                                                     const int o_stride_d) {
  extern __shared__ float shared_mem_cos_sin[];
  float *shared_mem_cos = shared_mem_cos_sin;
  float *shared_mem_sin = shared_mem_cos_sin + d2;
  int tid = threadIdx.x * blockDim.y + threadIdx.y;
  for (int i = tid; i < d2; i += blockDim.x * blockDim.y) {
    sincosf(freqs[s_id * d2 + i], &shared_mem_sin[i], &shared_mem_cos[i]);
  }
  __syncthreads();

#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
      float v_cos = shared_mem_cos[d_id];
      float v_sin = shared_mem_sin[d_id];
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate = v_src;  // ABLATION: no second load
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }
}

template <typename scalar_t>
__global__ void ablate_forward_kernel_no_partner_load(
    const scalar_t *src, const float *freqs, scalar_t *dst, const int s, const int h, const int d,
    const int d2, const int stride_s_or_t, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s_or_t, const int o_stride_b, const int o_stride_h,
    const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block = s_id * stride_s_or_t + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s_or_t + b_id * o_stride_b;
  ablate_block_forward_no_partner_load(src, freqs, dst, s_id, offset_block, offset_block_dst, h, d,
                                       d2, stride_h, stride_d, o_stride_h, o_stride_d);
}

// Reproduces TE's fused_rope_{forward,backward}_launcher for NVTE_BSHD, with the
// THD linear-grid branch removed (it is unreachable for BSHD).
template <bool BACKWARD>
torch::Tensor launch_bshd(const torch::Tensor &input, const torch::Tensor &freqs, bool interleaved,
                          const torch::Tensor *positions = nullptr) {
  TORCH_CHECK(input.dim() == 4, "expected (b, s, h, d) input");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16, "only bf16 is benchmarked here");
  TORCH_CHECK(freqs.scalar_type() == at::kFloat && freqs.is_contiguous(),
              "freqs must be contiguous float32");

  const int b = input.size(0);
  const int s = input.size(1);
  const int h = input.size(2);
  const int d = input.size(3);
  const int d2 = freqs.size(-1);
  TORCH_CHECK(d2 <= d, "freqs last dim must be <= head_dim");
  TORCH_CHECK(s <= freqs.numel() / d2,
              "seq_len (", s, ") exceeds freqs table rows (", freqs.numel() / d2,
              "); the kernel would read out of bounds");
  if (positions != nullptr) {
    TORCH_CHECK(positions->is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(positions->scalar_type() == at::kLong, "positions must be int64");
    TORCH_CHECK(positions->is_contiguous(), "positions must be contiguous");
    TORCH_CHECK(positions->dim() == 2 && positions->size(0) == b && positions->size(1) == s,
                "positions must have shape (batch, seq_len)");
  }

  auto output = torch::empty_like(input);

  const int stride_b = input.stride(0);
  const int stride_s_or_t = input.stride(1);
  const int stride_h = input.stride(2);
  const int stride_d = input.stride(3);

  // TE launcher, NVTE_BSHD branch.
  const int o_stride_s_or_t = h * d;
  const int o_stride_b = s * h * d;
  const int o_stride_h = d;
  const int o_stride_d = 1;

  int warps_per_block = h < 16 ? 4 : 8;
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  const int shared_mem_size = 2 * d2 * sizeof(float);  // cos, sin
  dim3 blocks(s, b);

  auto stream = at::cuda::getCurrentCUDAStream();
  const auto *src = reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr());
  auto *dst = reinterpret_cast<__nv_bfloat16 *>(output.data_ptr());
  const float *freqs_ptr = freqs.data_ptr<float>();

  if (BACKWARD && positions != nullptr) {
    fused_rope_backward_positions_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        src, positions->data_ptr<int64_t>(), freqs_ptr, dst, interleaved, s, h, d, d2,
        stride_s_or_t, stride_b, stride_h, stride_d, o_stride_s_or_t, o_stride_b, o_stride_h,
        o_stride_d);
  } else if (BACKWARD) {
    fused_rope_backward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        src, nullptr, freqs_ptr, nullptr, dst, interleaved, /*cp_size=*/1, /*cp_rank=*/0, s, h, d,
        d2, stride_s_or_t, stride_b, stride_h, stride_d, o_stride_s_or_t, o_stride_b, o_stride_h,
        o_stride_d);
  } else if (positions != nullptr) {
    fused_rope_forward_positions_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        src, positions->data_ptr<int64_t>(), freqs_ptr, dst, interleaved, s, h, d, d2,
        stride_s_or_t, stride_b, stride_h, stride_d, o_stride_s_or_t, o_stride_b, o_stride_h,
        o_stride_d);
  } else {
    fused_rope_forward_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        src, nullptr, freqs_ptr, nullptr, dst, interleaved, /*cp_size=*/1, /*cp_rank=*/0, s, h, d,
        d2, stride_s_or_t, stride_b, stride_h, stride_d, o_stride_s_or_t, o_stride_b, o_stride_h,
        o_stride_d);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

torch::Tensor te_rope_forward(const torch::Tensor &input, const torch::Tensor &freqs,
                              bool interleaved) {
  return launch_bshd<false>(input, freqs, interleaved);
}

torch::Tensor te_rope_backward(const torch::Tensor &grad_out, const torch::Tensor &freqs,
                               bool interleaved) {
  return launch_bshd<true>(grad_out, freqs, interleaved);
}

torch::Tensor te_rope_forward_positions(const torch::Tensor &input, const torch::Tensor &freqs,
                                        const torch::Tensor &positions, bool interleaved) {
  return launch_bshd<false>(input, freqs, interleaved, &positions);
}

torch::Tensor te_rope_backward_positions(const torch::Tensor &grad_out, const torch::Tensor &freqs,
                                         const torch::Tensor &positions, bool interleaved) {
  return launch_bshd<true>(grad_out, freqs, interleaved, &positions);
}

// DIAGNOSTIC ONLY -- see the ablation comment above. Results are incorrect.
torch::Tensor ablate_forward_no_partner_load(const torch::Tensor &input,
                                             const torch::Tensor &freqs) {
  const int b = input.size(0), s = input.size(1), h = input.size(2), d = input.size(3);
  const int d2 = freqs.size(-1);
  auto output = torch::empty_like(input);
  int warps_per_block = h < 16 ? 4 : 8;
  dim3 threads(THREADS_PER_WARP, warps_per_block);
  dim3 blocks(s, b);
  ablate_forward_kernel_no_partner_load<<<blocks, threads, 2 * d2 * sizeof(float),
                                          at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16 *>(input.data_ptr()), freqs.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()), s, h, d, d2, input.stride(1),
      input.stride(0), input.stride(2), input.stride(3), h * d, s * h * d, d, 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &te_rope_forward, "TE fused RoPE forward (BSHD, bf16)");
  m.def("backward", &te_rope_backward, "TE fused RoPE backward (BSHD, bf16)");
  m.def("forward_positions", &te_rope_forward_positions,
        "TE fused RoPE forward with per-token positions (BSHD, bf16)");
  m.def("backward_positions", &te_rope_backward_positions,
        "TE fused RoPE backward with per-token positions (BSHD, bf16)");
  m.def("ablate_no_partner_load", &ablate_forward_no_partner_load,
        "DIAGNOSTIC ONLY: TE forward with the rotate-half partner load removed");
}
