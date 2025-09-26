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

#include "launch_kernels_flash.h"


using namespace at;
using namespace torch;
using namespace gpu::xetla::attention;

std::vector<Tensor> flash_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    c10::optional<float> scale) {
  
  using T = bf16;
  constexpr gpu::xetla::gpu_arch arch_tag = gpu_arch::XeHpc;
  
  auto output = at::empty_like(query);
  auto debug_output = at::empty({query.size(0), query.size(1), query.size(2), key.size(2)}, 
                                at::kFloat).to(query.device());
  
  auto *query_ptr = query.data_ptr();
  auto *key_ptr = key.data_ptr();
  auto *value_ptr = value.data_ptr();
  auto *output_ptr = output.data_ptr();
  auto *debug_output_ptr = debug_output.data_ptr<float>();

  auto eclipse = dispatch_paged_attention_flash<T, arch_tag>(
      reinterpret_cast<T*>(query_ptr),
      reinterpret_cast<T*>(key_ptr),
      reinterpret_cast<T*>(value_ptr),
      reinterpret_cast<T*>(output_ptr),
      reinterpret_cast<float*>(debug_output_ptr),
      query.size(0), // num_batches
      query.size(1), // num_heads
      key.size(1), // num_kv_heads
      query.size(3), // head_size
      query.size(2), // num_queries
      key.size(2), // num_keys
      query.stride(0), // q_strideB
      query.stride(1), // q_strideN
      query.stride(2), // q_strideF
      key.stride(0), // kv_strideB
      key.stride(1), // kv_strideN
      key.stride(2), // kv_strideT
      scale.has_value() ? static_cast<float>(scale.value()) : 1.0f);

  return {output, debug_output};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &flash_attention, "Flash Attention");
}
