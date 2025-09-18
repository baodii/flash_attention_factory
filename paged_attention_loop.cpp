#include <iostream>
#include <stdexcept>
#include <optional>
#include <sycl/sycl.hpp>
#include <torch/torch.h>
#include <torch/extension.h>

#include "xetla.h"
#include "xetla_arch.h"

#include "c10/xpu/XPUFunctions.h"
#include "c10/xpu/XPUStream.h"

#include "launch_kernels_loop.h"


using namespace at;
using namespace torch;
using namespace gpu::xetla::attention;

std::vector<float> paged_attention_loop(
    torch::Tensor& max_logits,
    torch::Tensor& exp_sums,
    torch::Tensor& tem_output,
    torch::Tensor& output,
    torch::Tensor& debug_output,
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    const int64_t num_queries_per_tokens,
    const double sm_scale,
    const int64_t block_size,
    const int64_t max_context_len,
    const torch::Tensor& alibi_slopes,
    std::optional<torch::Tensor> &sinks,
    const double softcap) {
  
  using T = fp16;
  using U = uint32_t;
  constexpr gpu::xetla::gpu_arch arch_tag = gpu_arch::XeHpc;
  
  auto *query_ptr = query.data_ptr();
  auto *key_cache_ptr = key_cache.data_ptr();
  auto *value_cache_ptr = value_cache.data_ptr();
  auto *max_logits_ptr = max_logits.data_ptr<float>();
  auto *exp_sums_ptr = exp_sums.data_ptr<float>();
  auto *output_ptr = output.data_ptr();
  auto *tem_output_ptr = tem_output.data_ptr();
  auto *debug_output_ptr = debug_output.data_ptr<float>();
  auto *alibi_slopes_ptr = alibi_slopes.data_ptr<float>();
  auto *block_tables_ptr = block_tables.data_ptr();
  auto *context_lens_ptr = context_lens.data_ptr();
  void *sinks_ptr = nullptr;
  if (sinks.has_value()) {
    sinks_ptr = sinks.value().data_ptr();
  }

  uint32_t num_seqs = query.size(0);
  uint32_t num_heads = query.size(1);
  uint32_t head_size = query.size(2);
  uint32_t num_kv_heads = key_cache.size(2);
  uint32_t max_num_partitions = max_logits.size(2);
  uint32_t max_blocks_per_seq = block_tables.size(1);
    
  return dispatch_paged_attention_loop<T, U, arch_tag>(
      head_size, block_size, max_logits_ptr, exp_sums_ptr,
      reinterpret_cast<T *>(output_ptr),
      reinterpret_cast<T *>(tem_output_ptr), reinterpret_cast<T *>(query_ptr),
      reinterpret_cast<T *>(key_cache_ptr),
      reinterpret_cast<T *>(value_cache_ptr), 
      alibi_slopes_ptr,
      reinterpret_cast<T *>(sinks_ptr),
      reinterpret_cast<float *>(debug_output_ptr),
      reinterpret_cast<U *>(block_tables_ptr),
      reinterpret_cast<U *>(context_lens_ptr), max_num_partitions,
      num_queries_per_tokens, sm_scale, num_seqs, num_heads, num_kv_heads,
      max_blocks_per_seq, softcap);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &paged_attention_loop, "Paged Attention");
}
