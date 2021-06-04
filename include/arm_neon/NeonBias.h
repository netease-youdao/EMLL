/*****************************************************************************/
/* Copyright YouDao, Inc.                                                    */
/*                                                                           */
/* Licensed under the Apache License, Version 2.0 (the "License");           */
/* you may not use this file except in compliance with the License.          */
/* You may obtain a copy of the License at                                   */
/*                                                                           */
/*     http://www.apache.org/licenses/LICENSE-2.0                            */
/*                                                                           */
/* Unless required by applicable law or agreed to in writing, software       */
/* distributed under the License is distributed on an "AS IS" BASIS,         */
/* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  */
/* See the License for the specific language governing permissions and       */
/* limitations under the License.                                            */
/*****************************************************************************/


/*****************************************************************************
 * File:        NeonBias.h
 * Description: Bias functions based on ARM NEON instructions.
 ****************************************************************************/

#include <stdint.h>
#include <stdbool.h>
#include <arm_neon.h>

#ifndef INCLUDE_NEON_BIAS
#define INCLUDE_NEON_BIAS

/*****************************************************************************
 * Template:    NEON_BIAS
 * Description: Function template for NEON-based bias
 * Template Parameters: type_scalar: the type of scalar data,
 *                                   e.g. float for fp32 bias
 *                      type_vector: the type of SIMD vector data,
 *                                   e.g. float32x4_t
 *                      type_short: the short of data type in NEON intrinsics,
 *                                  e.g. f32 for fp32 bias
 *                      vector_size: the length of SIMD vector, e.g. 4 when
 *                                   type_vector == float32x4_t
 *                      fma: the short for multiply-add operation in the name
 *                           of NEON intrinsics. Use "fma" for fused
 *                           multiply-add and "mla" for sequential multiply-add
 * Function Parameters: C: the address of the matrix to apply bias on
 *                      bias_dim0: the bias value on every element
 *                      bias_dim1: the address of the input bias vector which
 *                                 will be applied to the matrix along its
 *                                 major dimension, i.e. when the element
 *                                 can be indexed by x * dim1 + y, each element
 *                                 is biased by bias_dim1[y]. No bias will be
 *                                 performed with NULL pointer as input.
 *                      bias_dim1_scale: the scale to be applied on elements
 *                                       of bias_dim1[] prior to the bias
 *                                       operation
 *                      bias_dim2: the address of the input bias vector which
 *                                 whill be applied to the matrix along its
 *                                 minor dimension, i.e. when the element
 *                                 can be indexed by x * dim1 + y, each element
 *                                 is biased by bias_dim2[x]. No bias will be
 *                                 performed with NULL pointer as input.
 *                      bias_dim2_scale: the scale to be applied on elements
 *                                       of bias_dim2[] prior to the bias
 *                                       operation
 *                      dim1: the length of the major dimension of input matrix
 *                      dim2: the length of the minor dimension of input matrix
 ****************************************************************************/
#define NEON_BIAS(type_scalar, type_vector, type_short, vector_size, fma) \
void bias_##type_scalar(type_scalar *C,\
  type_scalar bias_dim0,\
  const type_scalar *bias_dim1,\
  type_scalar bias_dim1_scale,\
  const type_scalar *bias_dim2,\
  type_scalar bias_dim2_scale,\
  uint32_t dim1, uint32_t dim2) {\
\
  bool do_bias_0 = (bias_dim0 != 0);\
  bool do_bias_1 = bias_dim1 && (bias_dim1_scale != 0);\
  bool do_bias_2 = bias_dim2 && (bias_dim2_scale != 0);\
\
  if (!do_bias_0 && !do_bias_1 && !do_bias_2) return;\
\
  if (!do_bias_1 && (do_bias_0 || do_bias_2)) {\
    type_scalar *c_ptr = C;\
    for (uint32_t dim2_pos = 0; dim2_pos < dim2; ++dim2_pos) {\
      const type_scalar bs = bias_dim0 + \
        (bias_dim2 ? bias_dim2[dim2_pos] * bias_dim2_scale : (type_scalar)0);\
      const type_vector bv = vdupq_n_##type_short(bs);\
      uint32_t dim1_left = dim1;\
      for (; dim1_left >= vector_size * 4; dim1_left -= vector_size * 4) {\
        type_vector c1 = vld1q_##type_short(c_ptr);\
        type_vector c2 = vld1q_##type_short(c_ptr + vector_size);\
        type_vector c3 = vld1q_##type_short(c_ptr + vector_size * 2);\
        type_vector c4 = vld1q_##type_short(c_ptr + vector_size * 3);\
        c1 = vaddq_##type_short(c1, bv);\
        c2 = vaddq_##type_short(c2, bv);\
        c3 = vaddq_##type_short(c3, bv);\
        c4 = vaddq_##type_short(c4, bv);\
        vst1q_##type_short(c_ptr, c1);\
        vst1q_##type_short(c_ptr + vector_size, c2);\
        vst1q_##type_short(c_ptr + vector_size * 2, c3);\
        vst1q_##type_short(c_ptr + vector_size * 3, c4);\
        c_ptr += vector_size * 4;\
      }\
      for (; dim1_left >= vector_size; dim1_left -= vector_size) {\
        type_vector c1 = vld1q_##type_short(c_ptr);\
        c1 = vaddq_##type_short(c1, bv);\
        vst1q_##type_short(c_ptr, c1); c_ptr += vector_size;\
      }\
      for (; dim1_left > 0; dim1_left--) {\
        *c_ptr += bs; c_ptr++;\
      }\
    }\
  } else if (do_bias_1 && !do_bias_0 && !do_bias_2) {\
    type_scalar *c_ptr = C;\
    for (uint32_t dim2_pos = 0; dim2_pos < dim2; ++dim2_pos) {\
      uint32_t dim1_left = dim1;\
      const type_scalar *bias_ptr = bias_dim1;\
      for (; dim1_left >= vector_size * 4; dim1_left -= vector_size * 4) {\
        type_vector c1 = vld1q_##type_short(c_ptr);\
        type_vector c2 = vld1q_##type_short(c_ptr + vector_size);\
        type_vector c3 = vld1q_##type_short(c_ptr + vector_size * 2);\
        type_vector c4 = vld1q_##type_short(c_ptr + vector_size * 3);\
        type_vector b1 = vld1q_##type_short(bias_ptr);\
        type_vector b2 = vld1q_##type_short(bias_ptr + vector_size);\
        type_vector b3 = vld1q_##type_short(bias_ptr + vector_size * 2);\
        type_vector b4 = vld1q_##type_short(bias_ptr + vector_size * 3);\
        bias_ptr += vector_size * 4;\
        c1 = v##fma##q_n_##type_short(c1, b1, bias_dim1_scale);\
        c2 = v##fma##q_n_##type_short(c2, b2, bias_dim1_scale);\
        c3 = v##fma##q_n_##type_short(c3, b3, bias_dim1_scale);\
        c4 = v##fma##q_n_##type_short(c4, b4, bias_dim1_scale);\
        vst1q_##type_short(c_ptr, c1);\
        vst1q_##type_short(c_ptr + vector_size, c2);\
        vst1q_##type_short(c_ptr + vector_size * 2, c3);\
        vst1q_##type_short(c_ptr + vector_size * 3, c4);\
        c_ptr += vector_size * 4;\
      }\
      for (; dim1_left >= vector_size; dim1_left -= vector_size) {\
        type_vector c1 = vld1q_##type_short(c_ptr);\
        type_vector b1 = vld1q_##type_short(bias_ptr);\
        bias_ptr += vector_size;\
        c1 = v##fma##q_n_##type_short(c1, b1, bias_dim1_scale);\
        vst1q_##type_short(c_ptr, c1);\
        c_ptr += vector_size;\
      }\
      for (; dim1_left > 0; dim1_left--) {\
        *c_ptr += (*bias_ptr) * bias_dim1_scale; bias_ptr++; c_ptr++;\
      }\
    }\
  } else {\
    type_scalar *c_ptr = C;\
    for (uint32_t dim2_pos = 0; dim2_pos < dim2; ++dim2_pos) {\
      const type_scalar bs = bias_dim0 + \
        (bias_dim2 ? bias_dim2[dim2_pos] * bias_dim2_scale : (type_scalar)0);\
      const type_vector bv = vdupq_n_##type_short(bs);\
      const type_scalar *bias_ptr = bias_dim1;\
      uint32_t dim1_left = dim1;\
      for (; dim1_left >= vector_size * 4; dim1_left -= vector_size * 4) {\
        type_vector c1 = vld1q_##type_short(c_ptr);\
        type_vector c2 = vld1q_##type_short(c_ptr + vector_size);\
        type_vector c3 = vld1q_##type_short(c_ptr + vector_size * 2);\
        type_vector c4 = vld1q_##type_short(c_ptr + vector_size * 3);\
        c1 = vaddq_##type_short(c1, bv);\
        c2 = vaddq_##type_short(c2, bv);\
        c3 = vaddq_##type_short(c3, bv);\
        c4 = vaddq_##type_short(c4, bv);\
        type_vector b1 = vld1q_##type_short(bias_ptr);\
        type_vector b2 = vld1q_##type_short(bias_ptr + vector_size);\
        type_vector b3 = vld1q_##type_short(bias_ptr + vector_size * 2);\
        type_vector b4 = vld1q_##type_short(bias_ptr + vector_size * 3);\
        bias_ptr += vector_size * 4;\
        c1 = v##fma##q_n_##type_short(c1, b1, bias_dim1_scale);\
        c2 = v##fma##q_n_##type_short(c2, b2, bias_dim1_scale);\
        c3 = v##fma##q_n_##type_short(c3, b3, bias_dim1_scale);\
        c4 = v##fma##q_n_##type_short(c4, b4, bias_dim1_scale);\
        vst1q_##type_short(c_ptr, c1);\
        vst1q_##type_short(c_ptr + vector_size, c2);\
        vst1q_##type_short(c_ptr + vector_size * 2, c3);\
        vst1q_##type_short(c_ptr + vector_size * 3, c4);\
        c_ptr += vector_size * 4;\
      }\
      for (; dim1_left >= vector_size; dim1_left -= vector_size) {\
        type_vector c1 = vld1q_##type_short(c_ptr);\
        c1 = vaddq_##type_short(c1, bv);\
        type_vector b1 = vld1q_##type_short(bias_ptr);\
        bias_ptr += vector_size;\
        c1 = v##fma##q_n_##type_short(c1, b1, bias_dim1_scale);\
        vst1q_##type_short(c_ptr, c1);\
        c_ptr += vector_size;\
      }\
      for (; dim1_left > 0; dim1_left--) {\
        *c_ptr += (*bias_ptr) * bias_dim1_scale + bs;\
        bias_ptr++; c_ptr++;\
      }\
    }\
  }\
}

#endif

