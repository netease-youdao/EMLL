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


#include "arm_neon/NeonIntOpSign.h"

#ifndef INCLUDE_I8I32DOT_COPY
#define INCLUDE_I8I32DOT_COPY

static inline void pref_ab(const I8 *dat) {
  __asm__ ("prfm pldl1keep,[%0,#64]\n\t"::"r"(dat):);
}

#define NCOPY_NEON_LOOP_K16_UNROLL4(inc, dst_ptr, src1, src2, src3, src4) \
  for (dim1_count = dim1; dim1_count > 15; dim1_count -= 16) {\
    I32X4X4 t1;\
    t1.val[0] = VREINTERPRETQ_I32_I8(VLD1Q_I8(src1));\
    src1 += 16; pref_ab(src1);\
    t1.val[1] = VREINTERPRETQ_I32_I8(VLD1Q_I8(src2));\
    src2 += 16; pref_ab(src2);\
    t1.val[2] = VREINTERPRETQ_I32_I8(VLD1Q_I8(src3));\
    src3 += 16; pref_ab(src3);\
    t1.val[3] = VREINTERPRETQ_I32_I8(VLD1Q_I8(src4));\
    src4 += 16; pref_ab(src4);\
    VST4Q_LANE_I32(dst_ptr, t1, 0);\
    VST4Q_LANE_I32(dst_ptr + inc, t1, 1);\
    VST4Q_LANE_I32(dst_ptr + inc * 2, t1, 2);\
    VST4Q_LANE_I32(dst_ptr + inc * 3, t1, 3);\
    dst_ptr += inc * 4;\
  }\
  if (dim1_count > 7) {\
    I32X2X4 t1;\
    t1.val[0] = VREINTERPRET_I32_I8(VLD1_I8(src1)); src1 += 8;\
    t1.val[1] = VREINTERPRET_I32_I8(VLD1_I8(src2)); src2 += 8;\
    t1.val[2] = VREINTERPRET_I32_I8(VLD1_I8(src3)); src3 += 8;\
    t1.val[3] = VREINTERPRET_I32_I8(VLD1_I8(src4)); src4 += 8;\
    VST4_LANE_I32(dst_ptr, t1, 0);\
    VST4_LANE_I32(dst_ptr + inc, t1, 1);\
    dst_ptr += inc * 2; dim1_count -= 8;\
  }\
  if (dim1_count > 3) {\
    __asm__(\
      "ldr w0,[%0],#4; ldr w1,[%1],#4; ldr w2,[%2],#4; ldr w3,[%3],#4\n\t"\
      "str w0,[%4]; str w1,[%4,#4]; str w2,[%4,#8]; str w3,[%4,#12]\n\t"\
      :"+r"(src1),"+r"(src2),"+r"(src3),"+r"(src4):"r"(dst_ptr)\
      :"cc","memory","x0","x1","x2","x3");\
    dst_ptr += inc; dim1_count -= 4;\
  }\
  if (dim1_count > 0) {\
    uint32_t *dst_cast = (uint32_t *)dst_ptr; dst_ptr += inc;\
    uint8_t *src1_cast = (uint8_t *)src1; src1 += dim1_count;\
    uint8_t *src2_cast = (uint8_t *)src2; src2 += dim1_count;\
    uint8_t *src3_cast = (uint8_t *)src3; src3 += dim1_count;\
    uint8_t *src4_cast = (uint8_t *)src4; src4 += dim1_count;\
    uint32_t d0, d1, d2, d3;\
    d0 = *src1_cast; d1 = *src2_cast;\
    d2 = *src3_cast; d3 = *src4_cast;\
    if (dim1_count >= 2) {\
      d0 |= ((uint32_t)src1_cast[1]) << 8;\
      d1 |= ((uint32_t)src2_cast[1]) << 8;\
      d2 |= ((uint32_t)src3_cast[1]) << 8;\
      d3 |= ((uint32_t)src4_cast[1]) << 8;\
    }\
    if (dim1_count >= 3) {\
      d0 |= ((uint32_t)src1_cast[2]) << 16;\
      d1 |= ((uint32_t)src2_cast[2]) << 16;\
      d2 |= ((uint32_t)src3_cast[2]) << 16;\
      d3 |= ((uint32_t)src4_cast[2]) << 16;\
    }\
    dst_cast[0] = d0; dst_cast[1] = d1;\
    dst_cast[2] = d2; dst_cast[3] = d3;\
  }

#define NCOPY_UNROLL_12 {\
  I32 *dst_h1 = dst1;\
  NCOPY_NEON_LOOP_K16_UNROLL4(12, dst_h1, src1, src2, src3, src4)\
  dst_h1 = dst1 + 4;\
  NCOPY_NEON_LOOP_K16_UNROLL4(12, dst_h1, src5, src6, src7, src8)\
  dst_h1 = dst1 + 8;\
  NCOPY_NEON_LOOP_K16_UNROLL4(12, dst_h1, src9, src10, src11, src12)\
  dst1 = dst_h1 - 8;\
}

#define NCOPY_UNROLL_8 {\
  I32 *dst_h1 = dst1;\
  NCOPY_NEON_LOOP_K16_UNROLL4(8, dst_h1, src1, src2, src3, src4)\
  dst_h1 = dst1 + 4;\
  NCOPY_NEON_LOOP_K16_UNROLL4(8, dst_h1, src5, src6, src7, src8)\
  dst1 = dst_h1 - 4;\
}

#define NCOPY_UNROLL_4 {\
  NCOPY_NEON_LOOP_K16_UNROLL4(4, dst1, src1, src2, src3, src4)\
}

#define NCOPY_UNROLL_2 {\
  for (dim1_count = dim1; dim1_count > 15; dim1_count -= 16) {\
    I32X4X2 t1;\
    t1.val[0] = VREINTERPRETQ_I32_I8(VLD1Q_I8(src1));\
    src1 += 16; pref_ab(src1);\
    t1.val[1] = VREINTERPRETQ_I32_I8(VLD1Q_I8(src2));\
    src2 += 16; pref_ab(src2);\
    VST2Q_I32(dst1, t1);\
    dst1 += 8;\
  }\
  if (dim1_count > 7) {\
    I32X2X2 t1;\
    t1.val[0] = VREINTERPRET_I32_I8(VLD1_I8(src1)); src1 += 8;\
    t1.val[1] = VREINTERPRET_I32_I8(VLD1_I8(src2)); src2 += 8;\
    VST2_I32(dst1, t1);\
    dst1 += 4; dim1_count -= 8;\
  }\
  if (dim1_count > 3) {\
    __asm__(\
      "ldr w0,[%0],#4; ldr w1,[%1],#4\n\t"\
      "str w0,[%2]; str w1,[%2,#4]\n\t"\
      :"+r"(src1),"+r"(src2):"r"(dst1)\
      :"cc","memory","x0","x1");\
    dst1 += 2; dim1_count -= 4;\
  }\
  if (dim1_count > 0) {\
    uint32_t *dst_cast = (uint32_t *)dst1; dst1 += 2;\
    uint8_t *src1_cast = (uint8_t *)src1; src1 += dim1_count;\
    uint8_t *src2_cast = (uint8_t *)src2; src2 += dim1_count;\
    uint32_t d0, d1;\
    d0 = *src1_cast; d1 = *src2_cast;\
    if (dim1_count >= 2) {\
      d0 |= ((uint32_t)src1_cast[1]) << 8;\
      d1 |= ((uint32_t)src2_cast[1]) << 8;\
    }\
    if (dim1_count >= 3) {\
      d0 |= ((uint32_t)src1_cast[2]) << 16;\
      d1 |= ((uint32_t)src2_cast[2]) << 16;\
    }\
    dst_cast[0] = d0; dst_cast[1] = d1;\
  }\
}

#define NCOPY_UNROLL_1 {\
  for (dim1_count = dim1; dim1_count > 15; dim1_count -= 16) {\
    I32X4 t1 = VREINTERPRETQ_I32_I8(VLD1Q_I8(src1));\
    src1 += 16;\
    VST1Q_I32(dst1, t1);\
    dst1 += 4;\
  }\
  if (dim1_count > 7) {\
    I32X2 t1 = VREINTERPRET_I32_I8(VLD1_I8(src1)); src1 += 8;\
    VST1_I32(dst1, t1);\
    dst1 += 2; dim1_count -= 8;\
  }\
  if (dim1_count > 3) {\
    __asm__(\
      "ldr w0,[%0],#4; str w0,[%1]\n\t"\
      :"+r"(src1):"r"(dst1)\
      :"cc","memory","x0","x1");\
    dst1++; dim1_count -= 4;\
  }\
  if (dim1_count > 0) {\
    uint32_t *dst_cast = (uint32_t *)dst1; dst1++;\
    uint8_t *src1_cast = (uint8_t *)src1; src1 += dim1_count;\
    uint32_t d0 = *src1_cast;\
    if (dim1_count >= 2) {\
      d0 |= ((uint32_t)src1_cast[1]) << 8;\
    }\
    if (dim1_count >= 3) {\
      d0 |= ((uint32_t)src1_cast[2]) << 16;\
    }\
    dst_cast[0] = d0;\
  }\
}

#ifdef GEMM_UNSIGNED_INT
#define NCOPY_uint8_t_uint32_t(unroll) NCOPY_UNROLL_##unroll
#else
#define NCOPY_int8_t_int32_t(unroll) NCOPY_UNROLL_##unroll
#endif

#define TCOPY_K4N8 {\
  uint8_t *src1_cast = (uint8_t *)src1; src1 += 8; pref_ab(src1);\
  uint8_t *src2_cast = (uint8_t *)src2; src2 += 8; pref_ab(src2);\
  uint8_t *src3_cast = (uint8_t *)src3; src3 += 8; pref_ab(src3);\
  uint8_t *src4_cast = (uint8_t *)src4; src4 += 8; pref_ab(src4);\
  uint8_t *dst1_cast = (uint8_t *)dst1; dst1 += 8;\
  uint8x8x4_t t1;\
  t1.val[0] = vld1_u8(src1_cast);\
  t1.val[1] = vld1_u8(src2_cast);\
  t1.val[2] = vld1_u8(src3_cast);\
  t1.val[3] = vld1_u8(src4_cast);\
  vst4_u8(dst1_cast, t1);\
}

#define TCOPY_K3N8 {\
  uint8_t *src1_cast = (uint8_t *)src1; src1 += 8; pref_ab(src1);\
  uint8_t *src2_cast = (uint8_t *)src2; src2 += 8; pref_ab(src2);\
  uint8_t *src3_cast = (uint8_t *)src3; src3 += 8; pref_ab(src3);\
  uint8_t *dst1_cast = (uint8_t *)dst1; dst1 += 8;\
  uint8x8x4_t t1;\
  t1.val[0] = vld1_u8(src1_cast);\
  t1.val[1] = vld1_u8(src2_cast);\
  t1.val[2] = vld1_u8(src3_cast);\
  t1.val[3] = vdup_n_u8(0);\
  vst4_u8(dst1_cast, t1);\
}

#define TCOPY_K2N8 {\
  uint8_t *src1_cast = (uint8_t *)src1; src1 += 8; pref_ab(src1);\
  uint8_t *src2_cast = (uint8_t *)src2; src2 += 8; pref_ab(src2);\
  uint8_t *dst1_cast = (uint8_t *)dst1; dst1 += 8;\
  uint8x8x4_t t1;\
  t1.val[0] = vld1_u8(src1_cast);\
  t1.val[1] = vld1_u8(src2_cast);\
  t1.val[2] = vdup_n_u8(0);\
  t1.val[3] = vdup_n_u8(0);\
  vst4_u8(dst1_cast, t1);\
}

#define TCOPY_K1N8 {\
  uint8_t *src1_cast = (uint8_t *)src1; src1 += 8;\
  uint8_t *dst1_cast = (uint8_t *)dst1; dst1 += 8;\
  uint8x8x4_t t1;\
  t1.val[0] = vld1_u8(src1_cast);\
  t1.val[1] = vdup_n_u8(0);\
  t1.val[2] = vdup_n_u8(0);\
  t1.val[3] = vdup_n_u8(0);\
  vst4_u8(dst1_cast, t1);\
}

#define LOAD_4_INCPTR_I8(ptr, v) \
  __asm__ __volatile__("ldr %s["#v"],[%["#ptr"]],#4\n\t"\
    :[v]"=w"(v),[ptr]"+r"(ptr)::"memory");

#define STORE_4X4_INTERLEAVE_I8(v1, v2, v3, v4, dst) \
  __asm__ __volatile__(\
    "zip1 %["#v1"].8b,%["#v1"].8b,%["#v2"].8b\n\t"\
    "zip1 %["#v3"].8b,%["#v3"].8b,%["#v4"].8b\n\t"\
    "zip1 %["#v1"].8h,%["#v1"].8h,%["#v3"].8h\n\t"\
    "str %q["#v1"],[%["#dst"]],#16\n\t"\
   :[v1]"+w"(v1), [v2]"+w"(v2), [v3]"+w"(v3), [v4]"+w"(v4), [dst]"+r"(dst)\
   ::"memory");

#define TCOPY_K4N4 {\
  I8X8 t1, t2, t3, t4;\
  LOAD_4_INCPTR_I8(src1, t1)\
  LOAD_4_INCPTR_I8(src2, t2)\
  LOAD_4_INCPTR_I8(src3, t3)\
  LOAD_4_INCPTR_I8(src4, t4)\
  STORE_4X4_INTERLEAVE_I8(t1, t2, t3, t4, dst1)\
}

#define TCOPY_K3N4 {\
  I8X8 t1, t2, t3, t4;\
  LOAD_4_INCPTR_I8(src1, t1)\
  LOAD_4_INCPTR_I8(src2, t2)\
  LOAD_4_INCPTR_I8(src3, t3)\
  t4 = VDUP_N_I8(0);\
  STORE_4X4_INTERLEAVE_I8(t1, t2, t3, t4, dst1)\
}

#define TCOPY_K2N4 {\
  I8X8 t1, t2, t3, t4;\
  LOAD_4_INCPTR_I8(src1, t1)\
  LOAD_4_INCPTR_I8(src2, t2)\
  t3 = VDUP_N_I8(0);\
  t4 = VDUP_N_I8(0);\
  STORE_4X4_INTERLEAVE_I8(t1, t2, t3, t4, dst1)\
}

#define TCOPY_K1N4 {\
  I8X8 t1, t2, t3, t4;\
  LOAD_4_INCPTR_I8(src1, t1)\
  t2 = VDUP_N_I8(0);\
  t3 = VDUP_N_I8(0);\
  t4 = VDUP_N_I8(0);\
  STORE_4X4_INTERLEAVE_I8(t1, t2, t3, t4, dst1)\
}

#define TCOPY_K4N2 \
  __asm__ __volatile__(\
    "ldr h0,[%0],#2; ldr h1,[%1],#2\n\t"\
    "ldr h2,[%2],#2; ldr h3,[%3],#2\n\t"\
    "st4 {v0.b,v1.b,v2.b,v3.b}[0],[%4],#4\n\t"\
    "st4 {v0.b,v1.b,v2.b,v3.b}[1],[%4],#4\n\t"\
    :"+r"(src1),"+r"(src2),"+r"(src3),"+r"(src4),"+r"(dst1)\
    ::"cc","memory","v0","v1","v2","v3");

#define TCOPY_K3N2 \
  __asm__ __volatile__(\
    "ldr h0,[%0],#2; ldr h1,[%1],#2\n\t"\
    "ldr h2,[%2],#2; movi v3.8b,#0\n\t"\
    "st4 {v0.b,v1.b,v2.b,v3.b}[0],[%3],#4\n\t"\
    "st4 {v0.b,v1.b,v2.b,v3.b}[1],[%3],#4\n\t"\
    :"+r"(src1),"+r"(src2),"+r"(src3),"+r"(dst1)\
    ::"cc","memory","v0","v1","v2","v3");

#define TCOPY_K2N2 \
  __asm__ __volatile__(\
    "ldr h0,[%0],#2; ldr h1,[%1],#2\n\t"\
    "movi v2.8b,#0; movi v3.8b,#0\n\t"\
    "st4 {v0.b,v1.b,v2.b,v3.b}[0],[%2],#4\n\t"\
    "st4 {v0.b,v1.b,v2.b,v3.b}[1],[%2],#4\n\t"\
    :"+r"(src1),"+r"(src2),"+r"(dst1)\
    ::"cc","memory","v0","v1","v2","v3");

#define TCOPY_K1N2 \
  __asm__ __volatile__(\
    "ldr h0,[%0],#2; movi v1.8b,#0\n\t"\
    "movi v2.8b,#0; movi v3.8b,#0\n\t"\
    "st4 {v0.b,v1.b,v2.b,v3.b}[0],[%1],#4\n\t"\
    "st4 {v0.b,v1.b,v2.b,v3.b}[1],[%1],#4\n\t"\
    :"+r"(src1),"+r"(dst1)\
    ::"cc","memory","v0","v1","v2","v3");

#define TCOPY_K4N1 \
  __asm__ __volatile__(\
    "ldr b0,[%0],#1; ldr b1,[%1],#1\n\t"\
    "ldr b2,[%2],#1; ldr b3,[%3],#1\n\t"\
    "st4 {v0.b,v1.b,v2.b,v3.b}[0],[%4]\n\t"\
    :"+r"(src1),"+r"(src2),"+r"(src3),"+r"(src4):"r"(dst1)\
    :"cc","memory","v0","v1","v2","v3");

#define TCOPY_K3N1 \
  __asm__ __volatile__(\
    "ldr b0,[%0],#1; ldr b1,[%1],#1\n\t"\
    "ldr b2,[%2],#1; movi v3.8b,#0\n\t"\
    "st4 {v0.b,v1.b,v2.b,v3.b}[0],[%3]\n\t"\
    :"+r"(src1),"+r"(src2),"+r"(src3):"r"(dst1)\
    :"cc","memory","v0","v1","v2","v3");

#define TCOPY_K2N1 \
  __asm__ __volatile__(\
    "ldr b0,[%0],#1; ldr b1,[%1],#1\n\t"\
    "movi v2.8b,#0; movi v3.8b,#0\n\t"\
    "st4 {v0.b,v1.b,v2.b,v3.b}[0],[%2]\n\t"\
    :"+r"(src1),"+r"(src2):"r"(dst1)\
    :"cc","memory","v0","v1","v2","v3");

#define TCOPY_K1N1 \
  __asm__ __volatile__(\
    "ldr b0,[%0],#1; str s0,[%1]\n\t"\
    :"+r"(src1):"r"(dst1)\
    :"cc","memory","v0");


#define TCOPY_NMAX12_TEMPLATE(kdim) \
  dst1 = dst + chunk_k_pass * 12;\
  for (; dim1_count > 11; dim1_count -= 12) {\
    TCOPY_K##kdim##N4 TCOPY_K##kdim##N8\
    dst1 += chunk_k_num * 12 - 12;\
  }\
  dst1 -= chunk_k_pass * 4;\
  if (dim1_count > 7) {\
    TCOPY_K##kdim##N8\
    dst1 += chunk_k_num * 8 - 8;\
    dim1_count -= 8;\
  }\
  dst1 -= chunk_k_pass * 4;\
  if (dim1_count > 3) {\
    TCOPY_K##kdim##N4\
    dst1 += chunk_k_num * 4 - 4;\
    dim1_count -= 4;\
  }\
  dst1 -= chunk_k_pass * 2;\
  if (dim1_count > 1) {\
    TCOPY_K##kdim##N2\
    dst1 += chunk_k_num * 2 - 2;\
    dim1_count -= 2;\
  }\
  dst1 -= chunk_k_pass;\
  if (dim1_count > 0) {\
    TCOPY_K##kdim##N1\
  }

#define TCOPY_NMAX8_TEMPLATE(kdim) \
  dst1 = dst + chunk_k_pass * 8;\
  for (; dim1_count > 7; dim1_count -= 8) {\
    TCOPY_K##kdim##N8\
    dst1 += chunk_k_num * 8 - 8;\
  }\
  dst1 -= chunk_k_pass * 4;\
  if (dim1_count > 3) {\
    TCOPY_K##kdim##N4\
    dst1 += chunk_k_num * 4 - 4;\
    dim1_count -= 4;\
  }\
  dst1 -= chunk_k_pass * 2;\
  if (dim1_count > 1) {\
    TCOPY_K##kdim##N2\
    dst1 += chunk_k_num * 2 - 2;\
    dim1_count -= 2;\
  }\
  dst1 -= chunk_k_pass;\
  if (dim1_count > 0) {\
    TCOPY_K##kdim##N1\
  }


#define TCOPY_FUNC_TEMPLATE(funcname, maxunroll) \
void funcname##maxunroll(\
  const I8 * __restrict__ src,\
  I32 * __restrict__ dst, uint32_t ld_dim,\
  uint32_t dim1, uint32_t dim2) {\
  if (!dim2) return;\
  uint32_t dim2_count = dim2;\
  const uint32_t chunk_k_num = ((dim2 - 1) >> 2) + 1;\
  const I8 *src0 = src;\
  for (; dim2_count > 3; dim2_count -= 4) {\
    const I8 *src1 = src0;\
    const I8 *src2 = src0 + ld_dim;\
    const I8 *src3 = src0 + ld_dim * 2;\
    const I8 *src4 = src0 + ld_dim * 3;\
    src0 += ld_dim * 4;\
    I32 *dst1;\
    uint32_t dim1_count = dim1;\
    const uint32_t chunk_k_pass = (dim2 - dim2_count) / 4;\
    TCOPY_NMAX##maxunroll##_TEMPLATE(4)\
  }\
  if (dim2_count == 3) {\
    const I8 *src1 = src0;\
    const I8 *src2 = src0 + ld_dim;\
    const I8 *src3 = src0 + ld_dim * 2;\
    I32 *dst1;\
    uint32_t dim1_count = dim1;\
    const uint32_t chunk_k_pass = chunk_k_num - 1;\
    TCOPY_NMAX##maxunroll##_TEMPLATE(3)\
  } else if (dim2_count == 2) {\
    const I8 *src1 = src0;\
    const I8 *src2 = src0 + ld_dim;\
    I32 *dst1;\
    uint32_t dim1_count = dim1;\
    const uint32_t chunk_k_pass = chunk_k_num - 1;\
    TCOPY_NMAX##maxunroll##_TEMPLATE(2)\
  } else if (dim2_count == 1) {\
    const I8 *src1 = src0;\
    I32 *dst1;\
    uint32_t dim1_count = dim1;\
    const uint32_t chunk_k_pass = chunk_k_num - 1;\
    TCOPY_NMAX##maxunroll##_TEMPLATE(1)\
  }\
}

#endif
