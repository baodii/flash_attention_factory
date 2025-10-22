#pragma once

#include "paged_attention_kernel_fp8_v2.hpp"
#include "paged_attention_policy_v2.hpp"

inline float get_exe_time(const sycl::event &e) {
  return (e.template get_profiling_info<
              sycl::info::event_profiling::command_end>() -
          e.template get_profiling_info<
              sycl::info::event_profiling::command_start>()) /
         1000.0f; // us
}

using namespace gpu::xetla::attention;

template <uint32_t head_size, uint32_t block_size, uint32_t query_group_size,
          typename T, typename U, fp8_format fp8_format_, gpu_arch arch_tag>
inline auto launch_paged_attn_fp8_v2(float *max_logits, float *exp_sums, T *out,
                                 T *tem_out, T *query, uint8_t *key_cache,
                                 uint8_t *value_cache, float *alibi_slopes, T *sinks,
                                 T *debug_out, U *block_tables, U *context_lens,
                                 uint32_t max_num_partitions, float sm_scale, float fp8_scale,
                                 uint32_t num_seqs, uint32_t num_heads,
                                 uint32_t num_kv_heads,
                                 uint32_t max_blocks_per_seq, float softcap) {
  using policy =
      paged_attention_policy_v2<head_size, block_size, query_group_size>;
  using kernel = paged_attention_kernel_fp8<policy, T, U, fp8_format_, arch_tag>;

  sycl::nd_range<3> nd_range =
      kernel::get_nd_range(num_seqs, num_kv_heads, max_num_partitions);

  /* auto queue = c10::xpu::getCurrentXPUStream().queue(); */
  auto propList =
      sycl::property_list{sycl::property::queue::enable_profiling()};
  sycl::queue q{sycl::gpu_selector_v, propList};
  auto event1 = q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<kernel>(
        nd_range, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
          kernel kernel_fn;
          typename kernel::arguments_t args(
              max_logits, exp_sums, reinterpret_cast<T *>(tem_out),
              reinterpret_cast<T *>(debug_out), reinterpret_cast<T *>(query),
              reinterpret_cast<uint8_t *>(key_cache),
              reinterpret_cast<uint8_t *>(value_cache),
              reinterpret_cast<T *>(sinks),
              reinterpret_cast<float *>(alibi_slopes),
              reinterpret_cast<U *>(block_tables),
              reinterpret_cast<U *>(context_lens), sm_scale, fp8_scale, num_seqs,
              num_heads, num_kv_heads, head_size, max_blocks_per_seq, softcap);
          kernel_fn(item, args);
        });
  });
  event1.wait();

  // launch reduce kernel

  using reduce_kernel = paged_attention_reduce<policy, T, U, arch_tag>;
  sycl::nd_range<3> nd_range_reduce =
      reduce_kernel::get_nd_range(num_seqs, num_heads);
  auto event2 = q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<reduce_kernel>(
        nd_range_reduce, [=](sycl::nd_item<3> item) SYCL_ESIMD_KERNEL {
          reduce_kernel reduce_kernel_fn;
          typename reduce_kernel::arguments_t args(
              reinterpret_cast<T *>(out), reinterpret_cast<T *>(tem_out),
              max_logits, exp_sums, reinterpret_cast<U *>(context_lens),
              num_seqs, num_heads, head_size, max_num_partitions);

          reduce_kernel_fn(item, args);
        });
  });
  event2.wait();
  return std::vector<float>{get_exe_time(event1),
                            get_exe_time(event2)}; // return execution times
}

template <typename T, typename U, fp8_format fp8_format_, gpu_arch arch_tag>
inline auto dispatch_paged_attention_fp8(
    uint32_t head_size, uint32_t block_size, float *max_logits, float *exp_sums,
    T *out, T *tem_out, T *query, uint8_t *key_cache, uint8_t *value_cache,
    float *alibi_slopes, T *sinks, T *debug_out, U *block_tables, U *context_lens,
    uint32_t max_num_partitions, uint32_t num_queries_per_tokens,
    float sm_scale, float fp8_scale, uint32_t num_seqs, uint32_t num_heads,
    uint32_t num_kv_heads, uint32_t max_blocks_per_seq, float softcap) {
  if (head_size == 128 && block_size == 64) {
    switch (num_queries_per_tokens) {
    case 1:
      return launch_paged_attn_fp8_v2<128, 64, 1, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 2:
      return launch_paged_attn_fp8_v2<128, 64, 2, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 3:
      return launch_paged_attn_fp8_v2<128, 64, 3, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 4:
      return launch_paged_attn_fp8_v2<128, 64, 4, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 5:
      return launch_paged_attn_fp8_v2<128, 64, 5, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 6:
      return launch_paged_attn_fp8_v2<128, 64, 6, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 7:
      return launch_paged_attn_fp8_v2<128, 64, 7, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 8:
      return launch_paged_attn_fp8_v2<128, 64, 8, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    default:
      throw std::runtime_error("Unsupported number of queries per token");
    }
  } else if (head_size == 64 && block_size == 64) {
    switch (num_queries_per_tokens) {
    case 1:
      return launch_paged_attn_fp8_v2<64, 64, 1, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 2:
      return launch_paged_attn_fp8_v2<64, 64, 2, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 3:
      return launch_paged_attn_fp8_v2<64, 64, 3, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 4:
      return launch_paged_attn_fp8_v2<64, 64, 4, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 5:
      return launch_paged_attn_fp8_v2<64, 64, 5, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 6:
      return launch_paged_attn_fp8_v2<64, 64, 6, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 7:
      return launch_paged_attn_fp8_v2<64, 64, 7, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    case 8:
      return launch_paged_attn_fp8_v2<64, 64, 8, T, U, fp8_format_, arch_tag>(
          max_logits, exp_sums, out, tem_out, query, key_cache, value_cache,
          alibi_slopes, sinks, debug_out, block_tables, context_lens,
          max_num_partitions, sm_scale, fp8_scale, num_seqs, num_heads, num_kv_heads,
          max_blocks_per_seq, softcap);
    default:
      throw std::runtime_error("Unsupported number of queries per token");
    }
  } else {
    throw std::runtime_error("Unsupported head size or block size");
  }
}

