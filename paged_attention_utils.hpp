/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in wriscalar_tg, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

#pragma once

#include <cmath>
#include <type_traits>
#include "common/core/common.hpp"
#include "subgroup/tile/api.hpp"
#include "xetla.hpp"

namespace gpu::xetla {

namespace attention {
constexpr float neg_infinity = INFINITY * -1;

template <typename mat_t>
struct tile_mask_t {
  using accum_t = typename mat_t::dtype;
  static constexpr accum_t kNegInfinity = INFINITY * -1;
  static constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  static constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  static constexpr uint32_t block_size_x = mat_t::block_size_x;
  static constexpr uint32_t block_size_y = mat_t::block_size_y;
  static constexpr int32_t num_block_x = mat_t::num_block_x;
  static constexpr uint32_t block_elems = mat_t::block_elems;

  // --------------------- // causal_mask // ---------------------- //

  inline static void causal_mask(
      mat_t& src,
      uint32_t start_x,
      uint32_t start_y) {
#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
      uint32_t blk_start_y = start_y + i * block_size_y;
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        uint32_t blk_start_x = start_x + j * block_size_x;
        if (blk_start_x + block_size_x > blk_start_y) {
          xetla_vector<uint32_t, block_size_x> blk_seq_x =
              xetla_vector_gen<uint32_t, block_size_x>(blk_start_x, 1);
          auto src_sub =
              src.reg
                  .xetla_select<block_elems, 1>(
                      (i * num_block_x + j) * block_elems)
                  .xetla_format<accum_t, block_size_y, block_size_x>();
#pragma unroll
          for (int k = 0; k < block_size_y; k++) {
            xetla_mask<block_size_x> mask = blk_seq_x > blk_start_y + k;
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }

    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr uint32_t tail_size_y = tile_size_y % block_size_y;
      constexpr uint32_t tail_block_elems = tail_size_y * block_size_x;

      uint32_t blk_start_y = start_y + tail_start_y;
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        uint32_t blk_start_x = start_x + j * block_size_x;
        if (blk_start_x + block_size_x > blk_start_y) {
          xetla_vector<uint32_t, block_size_x> blk_seq_x =
              xetla_vector_gen<uint32_t, block_size_x>(blk_start_x, 1);
          auto src_sub =
              src.reg
                  .xetla_select<tail_block_elems, 1>(
                      tail_start_y * tile_size_x + j * tail_block_elems)
                  .xetla_format<accum_t, tail_size_y, block_size_x>();
#pragma unroll
          for (int k = 0; k < tail_size_y; k++) {
            xetla_mask<block_size_x> mask = blk_seq_x > blk_start_y + k;
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }
  }

  // -------------------- // padding_mask // ---------------------- //

  inline static void padding_mask(mat_t& src, int num_keep) {
#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        int start_x = j * block_size_x;
        int num_keep_blk = std::max(0, num_keep - start_x);

        if (num_keep_blk < int(block_size_x)) {
          xetla_mask<block_size_x> mask =
              xetla_vector_gen<uint32_t, block_size_x>(1, 1) > num_keep_blk;
          auto src_sub =
              src.reg
                  .xetla_select<block_elems, 1>(
                      (i * num_block_x + j) * block_elems)
                  .xetla_format<accum_t, block_size_y, block_size_x>();
#pragma unroll
          for (uint32_t k = 0; k < block_size_y; k++) {
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }

    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr uint32_t tail_size_y = tile_size_y % block_size_y;
      constexpr uint32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        int start_x = j * block_size_x;
        int num_keep_blk = std::max(num_keep - start_x, 0);

        if (num_keep_blk < block_size_x) {
          xetla_mask<block_size_x> mask =
              xetla_vector_gen<uint32_t, block_size_x>(1, 1) > num_keep_blk;
          auto src_sub =
              src.reg
                  .xetla_select<tail_block_elems, 1>(
                      tail_start_y * tile_size_x + j * tail_block_elems)
                  .xetla_format<accum_t, tail_size_y, block_size_x>();
#pragma unroll
          for (int k = 0; k < tail_size_y; k++) {
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }
  }

  // -------------------- // local_mask // ---------------------- //

  inline static void local_mask(
      mat_t& src,
      uint32_t start_y,
      uint32_t start_x,
      int32_t w_left,
      int32_t w_right) {
#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
      int blk_start_y = start_y + i * block_size_y;
#pragma unroll
      for (uint32_t j = 0; j < num_block_x; j++) {
        auto src_sub = src.reg
                           .xetla_select<block_elems, 1>(
                               (i * num_block_x + j) * block_elems)
                           .xetla_format<accum_t, block_size_y, block_size_x>();
        int real_x = start_x + j * block_size_x;
        xetla_vector<int32_t, block_size_x> blk_seq_x =
            xetla_vector_gen<int32_t, block_size_x>(real_x, 1);
#pragma unroll
        for (uint32_t k = 0; k < block_size_y; k++) {
          int real_y = blk_start_y + k;
          if (w_right >= 0) {
            int local_right = real_y + w_right;
            xetla_mask<block_size_x> mask = blk_seq_x > local_right;
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }

          if (w_left >= 0) {
            int local_left = real_y - w_left;
            xetla_mask<block_size_x> mask = blk_seq_x < local_left;
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }

    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr uint32_t tail_size_y = tile_size_y % block_size_y;
      constexpr uint32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        auto src_sub =
            src.reg
                .xetla_select<tail_block_elems, 1>(
                    tail_start_y * tile_size_x + j * tail_block_elems)
                .xetla_format<accum_t, tail_size_y, block_size_x>();
        int real_x = start_x + j * block_size_x;
        xetla_vector<int32_t, block_size_x> blk_seq_x =
            xetla_vector_gen<int32_t, block_size_x>(real_x, 1);
#pragma unroll
        for (int k = 0; k < tail_size_y; k++) {
          int real_y = start_y + tail_start_y + k;
          if (w_right >= 0) {
            int local_right = real_y + w_right;
            xetla_mask<block_size_x> mask = blk_seq_x > local_right;
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }

          if (w_left >= 0) {
            int local_left = real_y - w_left;
            xetla_mask<block_size_x> mask = blk_seq_x < local_left;
            src_sub.row(k).xetla_merge(kNegInfinity, mask);
          }
        }
      }
    }
  }

  // -------------------- // alibi_mask // ---------------------- //

  inline static void alibi_mask(
      mat_t& src,
      float alibi_slopes,
      uint32_t start_x,
      [[maybe_unused]] uint32_t start_y) {
#pragma unroll
    for (uint32_t i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        uint32_t blk_start_x = start_x + j * block_size_x;
        xetla_vector<float, block_size_x> blk_seq_x =
            xetla_vector_gen<float, block_size_x>(blk_start_x, 1);
        auto src_sub = src.reg
                           .xetla_select<block_elems, 1>(
                               (i * num_block_x + j) * block_elems)
                           .xetla_format<accum_t, block_size_y, block_size_x>();
#pragma unroll
        for (int k = 0; k < block_size_y; k++) {
          src_sub.row(k) += (blk_seq_x * alibi_slopes);
        }
      }
    }

    if constexpr ((tile_size_y % block_size_y) != 0) {
      constexpr uint32_t tail_start_y =
          tile_size_y / block_size_y * block_size_y;
      constexpr uint32_t tail_size_y = tile_size_y % block_size_y;
      constexpr uint32_t tail_block_elems = tail_size_y * block_size_x;

#pragma unroll
      for (int j = 0; j < num_block_x; j++) {
        uint32_t blk_start_x = start_x + j * block_size_x;
        xetla_vector<float, block_size_x> blk_seq_x =
            xetla_vector_gen<float, block_size_x>(blk_start_x, 1);
        auto src_sub =
            src.reg
                .xetla_select<tail_block_elems, 1>(
                    tail_start_y * tile_size_x + j * tail_block_elems)
                .xetla_format<accum_t, tail_size_y, block_size_x>();
#pragma unroll
        for (int k = 0; k < tail_size_y; k++) {
          src_sub.row(k) += (blk_seq_x * alibi_slopes);
        }
      }
    }
  }
};

inline void SW_BARRIER() {
#if __INTEL_LLVM_COMPILER >= 20250000
#if defined(__SYCL_DEVICE_ONLY__)
  __asm__ volatile("fence_sw" : : :);
#endif // __SYCL_DEVICE_ONLY__
#else
  __ESIMD_NS::fence<__ESIMD_NS::fence_mask::sw_barrier>();
#endif // __INTEL_LLVM_COMPILER >= 20250000
}

// Matrix-Vector Multiplication, which computes the matrix vector product.
template <typename dtype, uint32_t N, typename mat_t, int dim>
inline typename std::enable_if_t<
    dim == 1 && std::is_same<dtype, typename mat_t::dtype>::value &&
        mat_t::tile_size_x == N,
    xetla_vector<dtype, mat_t::tile_size_y>>
mat_vec_mul(xetla_vector<dtype, N> vec, mat_t& mat) {
  constexpr uint32_t num_block_x = mat_t::num_block_x;
  constexpr uint32_t num_block_y = mat_t::num_block_y;
  constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  constexpr uint32_t block_size_x = mat_t::block_size_x;
  constexpr uint32_t block_size_y = mat_t::block_size_y;
  constexpr uint32_t block_elems = mat_t::block_elems;
  constexpr uint32_t remained_size_y = mat_t::remained_size_y;
  // The calculation process:
  // 1) allocate a temp buffer
  // 2) compute the product of m x v by each block
  // 3) accumulate the result into temp buffer
  // 4) reduce within the temp buffer
  xetla_vector<dtype, tile_size_y * block_size_x> acc;
#pragma unroll
  for (int i = 0; i < num_block_y; i++) {
    // j = 0
    {
      auto vec_sub = vec.xetla_select<block_size_x, 1>(0);
      auto mat_sub =
          mat.reg.xetla_select<block_elems, 1>(i * num_block_x * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();
#pragma unroll
      for (int row_i = 0; row_i < block_size_y; row_i++) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            (i * block_size_y + row_i) * block_size_x);
        acc_sub = vec_sub * mat_sub.row(row_i);
      }
    }
    // j = 1...
#pragma unroll
    for (int j = 1; j < num_block_x; j++) {
      auto vec_sub = vec.xetla_select<block_size_x, 1>(j * block_size_x);
      auto mat_sub =
          mat.reg
              .xetla_select<block_elems, 1>((i * num_block_x + j) * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();
#pragma unroll
      for (int row_i = 0; row_i < block_size_y; row_i++) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            (i * block_size_y + row_i) * block_size_x);
        acc_sub += vec_sub * mat_sub.row(row_i);
      }
    }
  }

  // process the tail
  if constexpr (remained_size_y != 0) {
    constexpr uint32_t remained_start_y = num_block_y * block_size_y;
    constexpr uint32_t remained_block_elems = remained_size_y * block_size_x;
    // j = 0
    {
      auto vec_sub = vec.xetla_select<block_size_x, 1>(0);
      auto mat_sub = mat.reg
                         .xetla_select<remained_block_elems, 1>(
                             remained_start_y * tile_size_x)
                         .xetla_format<dtype, remained_size_y, block_size_x>();
#pragma unroll
      for (int row_i = 0; row_i < remained_size_y; row_i++) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            (remained_start_y + row_i) * block_size_x);
        acc_sub = vec_sub * mat_sub.row(row_i);
      }
    }
    // j = 1...
#pragma unroll
    for (int j = 1; j < num_block_x; j++) {
      auto vec_sub = vec.xetla_select<block_size_x, 1>(j * block_size_x);
      auto mat_sub =
          mat.reg
              .xetla_select<remained_block_elems, 1>(
                  remained_start_y * tile_size_x + j * remained_block_elems)
              .xetla_format<dtype, remained_size_y, block_size_x>();
#pragma unroll
      for (int row_i = 0; row_i < block_size_y; row_i++) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            (remained_start_y + row_i) * block_size_x);
        acc_sub += vec_sub * mat_sub.row(row_i);
      }
    }
  }

  return recur_col_reduce<reduce_op::sum, dtype, block_size_x, tile_size_y>(
      acc);
}
template <typename dtype, uint32_t N, typename mat_t, int dim>
inline typename std::enable_if_t<
    dim == 0 && std::is_same<dtype, typename mat_t::dtype>::value &&
        mat_t::tile_size_y == N,
    xetla_vector<dtype, mat_t::tile_size_x>>
mat_vec_mul(xetla_vector<dtype, N> vec, mat_t& mat) {
  constexpr uint32_t num_block_x = mat_t::num_block_x;
  constexpr uint32_t num_block_y = mat_t::num_block_y;
  constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  constexpr uint32_t block_size_x = mat_t::block_size_x;
  constexpr uint32_t block_size_y = mat_t::block_size_y;
  constexpr uint32_t block_elems = mat_t::block_elems;
  constexpr uint32_t remained_size_y = mat_t::remained_size_y;

  using tile_desc_t = subgroup::
      tile_desc_t<tile_size_x, block_size_y, block_size_x, block_size_y>;
  using tile_t = subgroup::tile_t<dtype, tile_desc_t>;
  tile_t acc;

  for (int i = 0; i < num_block_x; ++i) {
    // j = 0
    auto mat_sub = mat.reg.xetla_select<block_elems, 1>(i * block_elems)
                       .xetla_format<dtype, block_size_y, block_size_x>();
    for (int row_i = 0; row_i < block_size_y; ++row_i) {
      auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
          i * block_elems + row_i * block_size_x);
      auto vec_sub = vec.xetla_select<1, 1>(row_i);
      acc_sub = vec_sub * mat_sub.row(row_i);
    }
  }

  for (int i = 0; i < num_block_x; ++i) {
    // j=1...
    for (int j = 1; j < num_block_y; ++j) {
      auto mat_sub =
          mat.reg
              .xetla_select<block_elems, 1>((j * num_block_x + i) * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();

      for (int row_i = 0; row_i < block_size_y; ++row_i) {
        auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
            i * block_elems + row_i * block_size_x);
        auto vec_sub = vec.xetla_select<1, 1>(j * block_size_y + row_i);
        acc_sub += vec_sub * mat_sub.row(row_i);
      }
    }
  }

  if constexpr (remained_size_y > 0) {
    for (int i = 0; i < num_block_x; ++i) {
      constexpr uint32_t remained_start_y = num_block_y * block_size_y;
      constexpr uint32_t remained_block_elems = remained_size_y * block_size_x;
      auto mat_sub =
          mat.reg
              .xetla_select<remained_block_elems, 1>(
                  remained_start_y * tile_size_x + i * remained_block_elems)
              .xetla_format<dtype, remained_size_y, block_size_x>();

      for (int row_i = 0; row_i < remained_size_y; ++row_i) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            i * block_elems + row_i * block_size_x);
        auto vec_sub = vec.xetla_select<1, 1>(remained_start_y + row_i);
        acc_sub += vec_sub * mat_sub.row(row_i);
      }
    }
  }

  auto res = tile_reduce<reduce_op::sum, dtype, dtype, 0>(acc);

  return res;
}
template <typename dtype, uint32_t N, typename mat_t, int dim>
inline typename std::enable_if_t<
    dim == 0 && std::is_same<dtype, typename mat_t::dtype>::value &&
        mat_t::tile_size_y == N,
    xetla_vector<dtype, mat_t::tile_size_x * mat_t::tile_size_y>>
mat_vec_mul_debug(xetla_vector<dtype, N> vec, mat_t& mat) {
  constexpr uint32_t num_block_x = mat_t::num_block_x;
  constexpr uint32_t num_block_y = mat_t::num_block_y;
  constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  constexpr uint32_t block_size_x = mat_t::block_size_x;
  constexpr uint32_t block_size_y = mat_t::block_size_y;
  constexpr uint32_t block_elems = mat_t::block_elems;
  constexpr uint32_t remained_size_y = mat_t::remained_size_y;

  using tile_desc_t = subgroup::
      tile_desc_t<tile_size_x, block_size_y, block_size_x, block_size_y>;
  using tile_t = subgroup::tile_t<dtype, tile_desc_t>;
  tile_t acc;

  for (int i = 0; i < num_block_x; ++i) {
    // j = 0
    auto mat_sub = mat.reg.xetla_select<block_elems, 1>(i * block_elems)
                       .xetla_format<dtype, block_size_y, block_size_x>();
    for (int row_i = 0; row_i < block_size_y; ++row_i) {
      auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
          i * block_elems + row_i * block_size_x);
      auto vec_sub = vec.xetla_select<1, 1>(row_i);
      acc_sub = vec_sub * mat_sub.row(row_i);
    }
  }

  for (int i = 0; i < num_block_x; ++i) {
    // j=1...
    for (int j = 1; j < num_block_y; ++j) {
      auto mat_sub =
          mat.reg
              .xetla_select<block_elems, 1>((j * num_block_x + i) * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();

      for (int row_i = 0; row_i < block_size_y; ++row_i) {
        auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
            i * block_elems + row_i * block_size_x);
        auto vec_sub = vec.xetla_select<1, 1>(j * block_size_y + row_i);
        acc_sub += vec_sub * mat_sub.row(row_i);
      }
    }
  }

  if constexpr (remained_size_y > 0) {
    for (int i = 0; i < num_block_x; ++i) {
      constexpr uint32_t remained_start_y = num_block_y * block_size_y;
      constexpr uint32_t remained_block_elems = remained_size_y * block_size_x;
      auto mat_sub =
          mat.reg
              .xetla_select<remained_block_elems, 1>(
                  remained_start_y * tile_size_x + i * remained_block_elems)
              .xetla_format<dtype, remained_size_y, block_size_x>();

      for (int row_i = 0; row_i < remained_size_y; ++row_i) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            i * block_elems + row_i * block_size_x);
        auto vec_sub = vec.xetla_select<1, 1>(remained_start_y + row_i);
        acc_sub += vec_sub * mat_sub.row(row_i);
      }
    }
  }

  auto res = tile_reduce<reduce_op::sum, dtype, dtype, 0>(acc);

  return acc.reg;
}

template <typename dtype, uint32_t N, typename mat_t, int dim>
inline typename std::enable_if_t<
    dim == 0 && std::is_same<dtype, typename mat_t::dtype>::value &&
        mat_t::tile_size_y == N,
    xetla_vector<dtype, mat_t::tile_size_x * mat_t::tile_size_y>>
mat_vec_mul_broadcast(xetla_vector<dtype, N> vec, mat_t& mat) {
  constexpr uint32_t num_block_x = mat_t::num_block_x;
  constexpr uint32_t num_block_y = mat_t::num_block_y;
  constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  constexpr uint32_t block_size_x = mat_t::block_size_x;
  constexpr uint32_t block_size_y = mat_t::block_size_y;
  constexpr uint32_t block_elems = mat_t::block_elems;
  constexpr uint32_t remained_size_y = mat_t::remained_size_y;

  using tile_desc_t = subgroup::
      tile_desc_t<tile_size_x, tile_size_y, block_size_x, block_size_y>;
  using tile_t = subgroup::tile_t<dtype, tile_desc_t>;
  tile_t acc;

  for (int i = 0; i < num_block_x; ++i) {
    // j = 0
    auto mat_sub = mat.reg.xetla_select<block_elems, 1>(i * block_elems)
                       .xetla_format<dtype, block_size_y, block_size_x>();
    for (int row_i = 0; row_i < block_size_y; ++row_i) {
      auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
          i * block_elems + row_i * block_size_x);
      auto vec_sub = vec.xetla_select<1, 1>(row_i);
      dtype vec_sub_data = vec_sub[0];
      sycl::ext::oneapi::experimental::printf("row_i: %d, vec_sub_data: %f\n", row_i, vec_sub_data);
      acc_sub = vec_sub * mat_sub.row(row_i);
    }
  }

  for (int i = 0; i < num_block_x; ++i) {
    // j=1...
    for (int j = 1; j < num_block_y; ++j) {
      auto mat_sub =
          mat.reg
              .xetla_select<block_elems, 1>((j * num_block_x + i) * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();

      for (int row_i = 0; row_i < block_size_y; ++row_i) {
        auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
            (j * num_block_x + i) * block_elems + row_i * block_size_x);
        auto vec_sub = vec.xetla_select<1, 1>(j * block_size_y + row_i);
        dtype vec_sub_data = vec_sub[0];
        sycl::ext::oneapi::experimental::printf("row_i: %d, vec_sub_data: %f\n", j * block_size_y + row_i, vec_sub_data);
        acc_sub = vec_sub * mat_sub.row(row_i);
      }
    }
  }

  if constexpr (remained_size_y > 0) {
    for (int i = 0; i < num_block_x; ++i) {
      constexpr uint32_t remained_start_y = num_block_y * block_size_y;
      constexpr uint32_t remained_block_elems = remained_size_y * block_size_x;
      auto mat_sub =
          mat.reg
              .xetla_select<remained_block_elems, 1>(
                  remained_start_y * tile_size_x + i * remained_block_elems)
              .xetla_format<dtype, remained_size_y, block_size_x>();

      for (int row_i = 0; row_i < remained_size_y; ++row_i) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            remained_start_y * tile_size_x + i * remained_block_elems + row_i * block_size_x);
        auto vec_sub = vec.xetla_select<1, 1>(remained_start_y + row_i);
        dtype vec_sub_data = vec_sub[0];
        sycl::ext::oneapi::experimental::printf("row_i: %d, vec_sub_data: %f\n", remained_start_y + row_i, vec_sub_data);
        acc_sub = vec_sub * mat_sub.row(row_i);
      }
    }
  }

  // auto res = tile_reduce<reduce_op::sum, dtype, dtype, 0>(acc);
  return acc.reg;
}

template <typename dtype, uint32_t N, typename mat_t, int dim>
inline typename std::enable_if_t<
    dim == 0 && std::is_same<dtype, typename mat_t::dtype>::value &&
        mat_t::tile_size_y == N,
    xetla_vector<dtype, mat_t::tile_size_x * mat_t::tile_size_y>>
mat_vec_div_broadcast(xetla_vector<dtype, N> vec, mat_t& mat) {
  constexpr uint32_t num_block_x = mat_t::num_block_x;
  constexpr uint32_t num_block_y = mat_t::num_block_y;
  constexpr uint32_t tile_size_x = mat_t::tile_size_x;
  constexpr uint32_t tile_size_y = mat_t::tile_size_y;
  constexpr uint32_t block_size_x = mat_t::block_size_x;
  constexpr uint32_t block_size_y = mat_t::block_size_y;
  constexpr uint32_t block_elems = mat_t::block_elems;
  constexpr uint32_t remained_size_y = mat_t::remained_size_y;

  using tile_desc_t = subgroup::
      tile_desc_t<tile_size_x, tile_size_y, block_size_x, block_size_y>;
  using tile_t = subgroup::tile_t<dtype, tile_desc_t>;
  tile_t acc;

  for (int i = 0; i < num_block_x; ++i) {
    // j = 0
    auto mat_sub = mat.reg.xetla_select<block_elems, 1>(i * block_elems)
                       .xetla_format<dtype, block_size_y, block_size_x>();
    for (int row_i = 0; row_i < block_size_y; ++row_i) {
      auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
          i * block_elems + row_i * block_size_x);
      auto vec_sub = vec.xetla_select<1, 1>(row_i);
      vec_sub = 1.0f / vec_sub;
      acc_sub = vec_sub * mat_sub.row(row_i);
    }
  }

  for (int i = 0; i < num_block_x; ++i) {
    // j=1...
    for (int j = 1; j < num_block_y; ++j) {
      auto mat_sub =
          mat.reg
              .xetla_select<block_elems, 1>((j * num_block_x + i) * block_elems)
              .xetla_format<dtype, block_size_y, block_size_x>();

      for (int row_i = 0; row_i < block_size_y; ++row_i) {
        auto acc_sub = acc.reg.xetla_select<block_size_x, 1>(
            i * block_elems + row_i * block_size_x + j * num_block_x * block_elems);
        auto vec_sub = vec.xetla_select<1, 1>(j * block_size_y + row_i);
        vec_sub = 1.0f / vec_sub;
        acc_sub = vec_sub * mat_sub.row(row_i);
      }
    }
  }

  if constexpr (remained_size_y > 0) {
    for (int i = 0; i < num_block_x; ++i) {
      constexpr uint32_t remained_start_y = num_block_y * block_size_y;
      constexpr uint32_t remained_block_elems = remained_size_y * block_size_x;
      auto mat_sub =
          mat.reg
              .xetla_select<remained_block_elems, 1>(
                  remained_start_y * tile_size_x + i * remained_block_elems)
              .xetla_format<dtype, remained_size_y, block_size_x>();

      for (int row_i = 0; row_i < remained_size_y; ++row_i) {
        auto acc_sub = acc.xetla_select<block_size_x, 1>(
            i * block_elems + row_i * block_size_x + remained_start_y * tile_size_x);
        auto vec_sub = vec.xetla_select<1, 1>(remained_start_y + row_i);
        vec_sub = 1.0f / vec_sub;
        acc_sub = vec_sub * mat_sub.row(row_i);
      }
    }
  }

  // auto res = tile_reduce<reduce_op::sum, dtype, dtype, 0>(acc);
  return acc.reg;
}

template <
    typename mat_t,
    uint32_t wg_size,
    reduce_op reduce_kind,
    gpu_arch arch_tag>
struct group_reduce_t {
  using dtype = typename mat_t::dtype;
  static constexpr uint32_t tile_size_x = mat_t::tile_size_x;

  // store results of subgroup to slm
  using local_st_tile_desc =
      subgroup::tile_desc_t<1, 1, 1, 1, reg_layout::tiled>;
  using local_st_tile_t = subgroup::tile_t<dtype, local_st_tile_desc>;
  using local_st_payload_t = subgroup::mem_payload_t<
      mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
      local_st_tile_desc,
      msg_type::block_1d,
      arch_tag>;
  // load all subgroup results together
  using local_ld_tile_desc =
      subgroup::tile_desc_t<wg_size, 1, wg_size, 1, reg_layout::tiled>;
  using local_ld_tile_t = subgroup::tile_t<dtype, local_ld_tile_desc>;
  using local_ld_payload_t = subgroup::mem_payload_t<
      mem_desc_t<dtype, mem_layout::row_major, mem_space::local>,
      local_ld_tile_desc,
      msg_type::block_1d,
      arch_tag>;
  // local variables
  xetla_nbarrier_t<wg_size, wg_size, arch_tag> nbarrier;
  uint32_t slm_base;
  uint32_t sg_id;
  uint32_t num_rows;
  inline group_reduce_t() = default;
  inline group_reduce_t(
      uint32_t num_rows_,
      uint32_t sg_id_,
      uint32_t nbarrier_id,
      uint32_t slm_base_) {
    nbarrier.init_nbarrier(nbarrier_id, nbarrier_role::producer_consumer);
    sg_id = sg_id_;
    slm_base = slm_base_;
    num_rows = num_rows_;
  }

  inline KERNEL_FUNC dtype operator()(mat_t& src) {
    // local reduction
    xetla_vector<dtype, 1> local_res(0);
    if constexpr (reduce_kind == reduce_op::max) {
      local_res[0] = neg_infinity;
    }

    if (num_rows > 0) {
      xetla_vector<dtype, tile_size_x> res =
          src.reg.xetla_select<tile_size_x, 1>(0);
      for (int row_i = 1; row_i < num_rows; row_i++) {
        res = reduce_helper<reduce_kind, dtype, tile_size_x>(
            res, src.reg.xetla_select<tile_size_x, 1>(row_i * tile_size_x));
      }
      local_res = recur_col_reduce<reduce_kind, dtype, tile_size_x, 1>(res);
    }

    if constexpr (wg_size == 1)
      return local_res[0];

    local_st_tile_t local_st;
    local_st_payload_t local_st_payload(
        slm_base, wg_size, 1, wg_size, sg_id, 0);
    local_st.reg = local_res;
    subgroup::tile_store(local_st, local_st_payload);

    xetla_fence<memory_kind::shared_local>();
    nbarrier.arrive_wait();

    local_ld_tile_t local_ld;
    local_ld_payload_t local_ld_payload(slm_base, wg_size, 1, wg_size, 0, 0);
    subgroup::tile_load(local_ld, local_ld_payload);

    return xetla_reduce<dtype, dtype, wg_size, reduce_kind>(local_ld.reg);
  }
};

} // namespace attention

} // namespace gpu::xetla
