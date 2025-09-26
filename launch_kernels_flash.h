#pragma once

#include "fmha_forward.hpp"
#include "fmha_forward_policy.h"

inline float get_exe_time(const sycl::event &e) {
  return (e.template get_profiling_info<
              sycl::info::event_profiling::command_end>() -
          e.template get_profiling_info<
              sycl::info::event_profiling::command_start>()) /
         1000.0f; // us
}

using namespace gpu::xetla::attention;

template <typename T, gpu_arch arch_tag>
inline auto dispatch_paged_attention_flash(
    T *query_ptr, T *key_ptr, T *value_ptr, T *out_ptr, float *debug_ptr,
    uint32_t num_batches,
    uint32_t num_heads, uint32_t num_kv_heads, uint32_t head_size,
    uint32_t num_queries, uint32_t num_keys, uint32_t q_strideB,
    uint32_t q_strideN, uint32_t q_strideF, uint32_t kv_strideB,
    uint32_t kv_strideN, uint32_t kv_strideT, float softmax_scale) {
  using policy = fmha_policy_64x128x128;
  using kernel = fmha_forward_t<policy, T, arch_tag, false, false, false, false,
                                false, false, false, false>;

  auto nd_range = kernel::get_nd_range(num_batches * num_heads, num_queries);
  auto propList =
      sycl::property_list{sycl::property::queue::enable_profiling()};
  sycl::queue q{sycl::gpu_selector_v, propList};
  auto event = q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel>(
        nd_range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
          kernel kernel_fn;
          typename kernel::arguments_t args(
              query_ptr, 
              key_ptr, 
              value_ptr,
              nullptr, // alibi
              nullptr, // sink
              nullptr, // attn_mask
              nullptr, // dropout_mask
              out_ptr,
              debug_ptr,
              nullptr, // softmax_lse
              num_batches, 
              num_heads, 
              num_kv_heads, 
              head_size, 
              num_queries,
              num_keys, 
              q_strideB, 
              q_strideN, 
              q_strideF, 
              kv_strideB, 
              kv_strideN,
              kv_strideT, 
              -1, 
              -1, 
              -1, 
              nullptr, 
              nullptr, 
              softmax_scale,
              0,  // dropout_prob
              0,  // alibi_padded_block_size
              0,  // attn_mask_padded_block_size
              -1, // window_size_left
              -1, // window_size_right
              -1, // seed_t,
              -1, // offset_t
              -1, 
              nullptr, 
              0, 
              0);
          kernel_fn(item, args);
        });
  });
  event.wait();
  return std::vector<float>{get_exe_time(event)};
}
