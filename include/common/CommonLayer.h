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


/******************************************************************************
 * File:        CommonLayer.h
 * Description: Function templates for operations in neural network layers
 *****************************************************************************/

#include <stdlib.h>
#include <stdbool.h>

#ifndef INCLUDE_COMMON_LAYER
#define INCLUDE_COMMON_LAYER

/* function template for fully-connected layer, serial & OpenMP */
#define SIMPLE_FC_FUNC(gemmtype, wtype, itype, otype, ...) \
int fc##__VA_ARGS__(const itype *src, const wtype *weight,\
  const otype *bias, otype *output, int M, int K, int N,\
  int trans_src, int trans_weight, int num_threads) {\
\
  int status = gemmtype(trans_weight, trans_src,\
    weight, src, output, N, M, K, 0, num_threads);\
  if (status) return status;\
  bias_##otype(output, 0.0, bias, 1.0, NULL, 0.0, N, M);\
  return status;\
}

/* function template for bias layer */
#define STD_BIAS_FUNC(type) \
void bias_##type(type *C, type bias_dim0, const type *bias_dim1,\
  type bias_dim1_scale, const type *bias_dim2, type bias_dim2_scale,\
  uint32_t dim1, uint32_t dim2) {\
\
  if (!C) return;\
\
  bool do_bias_0 = (bias_dim0 != 0);\
  bool do_bias_1 = bias_dim1 && (bias_dim1_scale != 0);\
  bool do_bias_2 = bias_dim2 && (bias_dim2_scale != 0);\
\
  if (!do_bias_0 && !do_bias_1 && !do_bias_2) return;\
\
  if (!do_bias_1 && (do_bias_0 || do_bias_2)) {\
    for (uint32_t dim2_pos = 0; dim2_pos < dim2; ++dim2_pos) {\
      type *c_ptr = C + dim2_pos * dim1;\
      const type bs = bias_dim0 + \
        (bias_dim2 ? bias_dim2[dim2_pos] * bias_dim2_scale : 0);\
      _Pragma("omp simd")\
      for (uint32_t dim1_pos = 0; dim1_pos < dim1; ++dim1_pos) {\
        c_ptr[dim1_pos] += bs;\
      }\
    }\
  } else if (do_bias_1 && !do_bias_0 && !do_bias_2) {\
    for (uint32_t dim2_pos = 0; dim2_pos < dim2; ++dim2_pos) {\
      type *c_ptr = C + dim2_pos * dim1;\
      const type *bias_ptr = bias_dim1;\
      _Pragma("omp simd")\
      for (uint32_t dim1_pos = 0; dim1_pos < dim1; ++dim1_pos) {\
        c_ptr[dim1_pos] += bias_ptr[dim1_pos] * bias_dim1_scale;\
      }\
    }\
  } else {\
    for (uint32_t dim2_pos = 0; dim2_pos < dim2; ++dim2_pos) {\
      type *c_ptr = C + dim2_pos * dim1;\
      const type bs = bias_dim0 + \
        (bias_dim2 ? bias_dim2[dim2_pos] * bias_dim2_scale : 0);\
      const type *bias_ptr = bias_dim1;\
      _Pragma("omp simd")\
      for (uint32_t dim1_pos = 0; dim1_pos < dim1; ++dim1_pos) {\
        c_ptr[dim1_pos] += bs +\
          bias_ptr[dim1_pos] * bias_dim1_scale;\
      }\
    }\
  }\
}

#endif
