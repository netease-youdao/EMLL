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
 * File:        CommonTest.h
 * Description: Common test framework for GEMM/Bias/Quantization functions
 * Usage:       Include this header, then define test functions by macros,
 *              last call test functions in main function. Please refer to
 *              test/Test*.c for example.
 *****************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

#ifndef INCLUDE_COMMON_TEST
#define INCLUDE_COMMON_TEST

#define STD_GEMM_AR_BC_CC(atype, btype, ctype, A, B, C, M, N, K, beta) {\
  for (uint32_t n_pos = 0; n_pos < (N); ++n_pos) {\
    ctype *c_ptr = (C) + n_pos * (M);\
    const btype *b_ptr = (B) + n_pos * (K);\
    for (uint32_t m_pos = 0; m_pos < (M); ++m_pos) {\
      const atype *a_ptr = (A) + m_pos * (K);\
      ctype sum = (ctype)0.0f;\
      for (uint32_t k_pos = 0; k_pos < (K); ++k_pos) {\
        sum += (ctype)a_ptr[k_pos] * (ctype)b_ptr[k_pos];\
      }\
      c_ptr[m_pos] = c_ptr[m_pos] * beta + sum;\
    }\
  }\
}

/* src: row-major; dst: column-major */
#define STD_TRANSPOSE(T, src, dst, src_rows, src_cols) {\
  for (uint32_t src_row_pos = 0; src_row_pos < src_rows; ++src_row_pos) {\
    const T *src_ptr = src + src_row_pos * src_cols;\
    T *dst_ptr = dst + src_row_pos;\
    for (uint32_t src_col_pos = 0; src_col_pos < src_cols; ++src_col_pos) {\
      dst_ptr[src_col_pos * src_rows] = src_ptr[src_col_pos];\
    }\
  }\
}

/* matrix C is column-major */
#define STD_GEMM(gemmtype, atype, btype, ctype) \
void std_##gemmtype(const atype *A, const btype *B, ctype *C,\
  uint32_t M, uint32_t N, uint32_t K,\
  bool a_rowmajor, bool b_colmajor, ctype beta) {\
  atype *A_mat = NULL; const atype *A_rd = A;\
  if (!a_rowmajor) {\
    A_mat = (atype *)malloc(M * K * sizeof(atype));\
    STD_TRANSPOSE(atype, A, A_mat, K, M)\
    A_rd = A_mat;\
  }\
  btype *B_mat = NULL; const btype *B_rd = B;\
  if (!b_colmajor) {\
    B_mat = (btype *)malloc(N * K * sizeof(btype));\
    STD_TRANSPOSE(btype, B, B_mat, K, N)\
    B_rd = B_mat;\
  }\
  STD_GEMM_AR_BC_CC(atype, btype, ctype, A_rd, B_rd, C, M, N, K, beta)\
  if (A_mat) free(A_mat);\
  if (B_mat) free(B_mat);\
}

/* produce a random number from a/b, a is a random number in [-c, c] */
/* c = dividend_abs_max; b = divisor */
#define STD_RAND(T, dat, size, dividend_abs_max, divisor) {\
  const int32_t abs_max_get = (dividend_abs_max) < 0 ? \
    -(dividend_abs_max) : (dividend_abs_max);\
  const int32_t offset_get = (dividend_abs_max) < 0 ? \
    0 : (dividend_abs_max);\
  for (uint64_t pos = 0; pos < (size); ++pos) {\
    int32_t rand_i = rand() % (2 * abs_max_get + 1);\
    rand_i -= offset_get;\
    float rand_f = (float)rand_i / (float)(divisor);\
    *((dat) + pos) = (T)rand_f;\
  }\
}

#define STD_MAXDIFF(T, max, dat1, dat2, size) {\
  T tmp;\
  max = (T)0.0f;\
  for (uint64_t pos = 0; pos < (size); ++pos) {\
    tmp = (*((dat2) + pos)) - (*((dat1) + pos));\
    if (tmp < 0) tmp = (T)0.0f - tmp;\
    if (tmp > max) max = tmp;\
  }\
}

#define SRC_SIZE 160000000

#define STD_TEST(gemmtype, btype, atype, ctype, dividend_abs_max, divisor) \
STD_GEMM(gemmtype, atype, btype, ctype)\
typedef int (*TestFunc_##gemmtype)(int, int, const atype*, const btype*, ctype*,\
  uint32_t, uint32_t, uint32_t, ctype, uint32_t);\
void std_test_##gemmtype(TestFunc_##gemmtype test_gemm,\
  uint32_t M, uint32_t N, uint32_t K, uint8_t transAB,\
  ctype beta, uint32_t num_threads) {\
\
  const int b_rowmajor = transAB & 2;\
  const int a_rowmajor = transAB & 1;\
\
  const uint64_t a_size = (uint64_t)M * (uint64_t)K;\
  const uint64_t b_size = (uint64_t)N * (uint64_t)K;\
  const uint64_t c_size = (uint64_t)M * (uint64_t)N;\
  const uint64_t iters = (uint64_t)SRC_SIZE / \
    (a_size * sizeof(atype) + b_size * sizeof(btype) + 1);\
  if (iters == 0) {\
    printf("Problem size too large. return.\n");\
    return;\
  }\
  atype * const A = (atype *)malloc(a_size * iters * sizeof(atype));\
  btype * const B = (btype *)malloc(b_size * iters * sizeof(btype));\
  ctype * const C_ref = (ctype *)malloc(c_size * sizeof(ctype));\
  ctype * const C_tst = (ctype *)malloc(c_size * sizeof(ctype));\
  if (A == NULL || B == NULL || C_ref == NULL || C_tst == NULL) {\
    printf("Memory allocation failed. return.\n");\
    free(A); free(B); free(C_ref); free(C_tst);\
    return;\
  }\
  srand(time(NULL));\
  STD_RAND(float, A, a_size, dividend_abs_max, divisor)\
  for (uint64_t pos = 1; pos < iters; ++pos) {\
    memcpy(A + pos * a_size, A, a_size * sizeof(atype));\
  }\
  STD_RAND(float, B, b_size, dividend_abs_max, divisor)\
  for (uint64_t pos = 1; pos < iters; ++pos) {\
    memcpy(B + pos * b_size, B, b_size * sizeof(btype));\
  }\
  STD_RAND(float, C_tst, c_size, dividend_abs_max, divisor)\
  memcpy(C_ref, C_tst, c_size * sizeof(ctype));\
  struct timespec st, et;\
  std_##gemmtype(A, B, C_ref, M, N, K, a_rowmajor, !b_rowmajor, beta);\
  clock_gettime(CLOCK_MONOTONIC, &st);\
  int ret_status = test_gemm(a_rowmajor, b_rowmajor, A, B, C_tst,\
    M, N, K, beta, num_threads);\
  clock_gettime(CLOCK_MONOTONIC, &et);\
  double nsec = (double)(et.tv_nsec - st.tv_nsec) + 1.0e9 * \
    (double)(et.tv_sec - st.tv_sec);\
  printf("Time elapsed for the first run: %.2e ns\n", nsec);\
  if (ret_status) {\
    printf("An error has occurred in the tested gemm, error code = %d\n",\
      ret_status);\
    return;\
  }\
  ctype max;\
  STD_MAXDIFF(float, max, C_ref, C_tst, c_size)\
  printf("Max diff. between test and std: %.2e\n", (double)max);\
\
  if (iters > 1) {\
    clock_gettime(CLOCK_MONOTONIC, &st);\
    for (uint64_t pos = 1; pos < iters; ++pos) {\
      test_gemm(a_rowmajor, b_rowmajor, A + a_size * pos, B + b_size * pos, C_tst,\
        M, N, K, -1, num_threads);\
    }\
    clock_gettime(CLOCK_MONOTONIC, &et);\
    double nsec = (double)(et.tv_nsec - st.tv_nsec) + 1.0e9 * \
      (double)(et.tv_sec - st.tv_sec);\
    double ops = (double)M * (double)N * (double)(2 * K - 1) * \
      (double)(iters - 1);\
    printf("Averaged time for each run after warm-up: %.2e ns\n",\
      nsec / (double)(iters - 1));\
    printf("The performance of test: %.2e GFLOPS\n", ops / nsec);\
  }\
\
  free(A); free(B); free(C_ref); free(C_tst);\
  return;\
}

#define TEST_1D_OPERATION_PERF(size, num_iters, FUNC_CALLER, ...) \
  struct timespec st, et;\
  clock_gettime(CLOCK_MONOTONIC, &st);\
  for (uint32_t pos = 1; pos < num_iters; ++pos) {\
    FUNC_CALLER(0, size, ##__VA_ARGS__)\
  }\
  clock_gettime(CLOCK_MONOTONIC, &et);\
  double nsec = (double)(et.tv_nsec - st.tv_nsec) + 1.0e9 * (double)\
    (et.tv_sec - st.tv_sec);\
  printf("Avg. Perf.(repeat on the same data): %.2e G elements per second\n",\
    (double)size * (double)(num_iters - 1) / nsec);\
  clock_gettime(CLOCK_MONOTONIC, &st);\
  for (uint32_t pos = 1; pos < num_iters; ++pos) {\
    FUNC_CALLER(pos, size, ##__VA_ARGS__)\
  }\
  clock_gettime(CLOCK_MONOTONIC, &et);\
  nsec = (double)(et.tv_nsec - st.tv_nsec) + 1.0e9 * (double)\
    (et.tv_sec - st.tv_sec);\
  printf("Avg. Perf.(no repeat of data region): %.2e G elements per second\n",\
    (double)size * (double)(num_iters - 1) / nsec);

#define FUNC_CALLER_QUANT_UNSYM(pos, size, inbits, outbits,\
  src, tst_u, zero_addr, scale_addr) \
  quantize_asymmetric_f##inbits##_u##outbits(\
    src + pos * size, tst_u + pos * size,\
    zero_addr, scale_addr, size, 0, -1);

#define TEST_QUANT_UNSYM(inbits, outbits) \
static void test_quant_asym_f##inbits##_u##outbits(uint32_t size) {\
  if (size < 4) size = 4;\
  printf("Test unsymmetrical quantization fp"#inbits" -> uint"#outbits":\n");\
  printf("num_elements = %u\n", size);\
\
  const uint32_t num_iters = 40000000 / size;\
  if (num_iters <= 2) {\
    printf("Problem size too large.\n");\
    return;\
  }\
\
  uint##outbits##_t * const ref_u =\
    (uint##outbits##_t *)malloc(size * (outbits >> 3));\
  uint##outbits##_t * const tst_u =\
    (uint##outbits##_t *)malloc(num_iters * size * (outbits >> 3));\
  float##inbits##_t * const src =\
    (float##inbits##_t *)malloc(num_iters * size * (inbits >> 3));\
\
  srand(time(NULL));\
  for (uint32_t pos = 0; pos < size; ++pos) {\
    ref_u[pos] = rand();\
  }\
  uint32_t min_pos = rand() % size;\
  uint32_t max_pos = min_pos;\
  while (max_pos == min_pos) {\
    max_pos = rand() % size;\
  }\
  ref_u[min_pos] = 0;\
  ref_u[max_pos] = (uint##outbits##_t)-1;\
  const float##inbits##_t ref_scale =\
    (float##inbits##_t)(rand() + 1) / (float##inbits##_t)(RAND_MAX >> 2);\
  const uint##outbits##_t ref_zero = rand();\
  printf("Generate src data with ref_zero = %u and ref_scale = %.2e\n",\
    ref_zero, ref_scale);\
  for (uint32_t pos = 0; pos < size; ++pos) {\
    float##inbits##_t fluc =\
      ((float##inbits##_t)rand() / RAND_MAX - (float##inbits##_t)0.5) *\
      (float##inbits##_t)0.9875;\
    if (pos == max_pos || pos == min_pos) fluc = 0.0;\
    else if (ref_u[pos] == (uint##outbits##_t)-1 && fluc > 0) fluc *= -1.0;\
    else if (ref_u[pos] == 0 && fluc < 0) fluc *= -1.0;\
    src[pos] = ((float##inbits##_t)((long)ref_u[pos] - (long)ref_zero) + fluc)\
      * ref_scale;\
  }\
  printf("First 4 elements of ref_u"#outbits"\n: %u, %u, %u, %u\n",\
    ref_u[0], ref_u[1], ref_u[2], ref_u[3]);\
  printf("First 4 elements of src_f"#inbits"\n: %.2e, %.2e, %.2e, %.2e\n",\
    src[0], src[1], src[2], src[3]);\
  for (uint32_t pos = 1; pos < num_iters; ++pos) {\
    memcpy(src + pos * size, src, size * (inbits >> 3));\
  }\
\
  uint##outbits##_t tst_zero;\
  float##inbits##_t tst_scale;\
  quantize_asymmetric_f##inbits##_u##outbits(\
    src, tst_u, &tst_zero, &tst_scale, size, 0, -1);\
\
  if (tst_zero != ref_zero) {\
    printf("tst_zero = %u, mismatch with ref_zero\n", tst_zero);\
  }\
  printf("relative difference between ref_scale and tst_scale: %.2e\n",\
    (tst_scale - ref_scale) / ref_scale);\
  int eql = 1;\
  for (uint32_t pos = 0; pos < size; ++pos) {\
    if (eql != 0 && tst_u[pos] != ref_u[pos]) {\
      eql = 0;\
      printf("u"#outbits" results at pos %u are inconsistent: ref = %u, tst = %u\n",\
        pos, ref_u[pos], tst_u[pos]);\
      break;\
    }\
  }\
  if (eql != 0) {\
    printf("u"#outbits" results are equal\n");\
    TEST_1D_OPERATION_PERF(size, num_iters, FUNC_CALLER_QUANT_UNSYM,\
      inbits, outbits, src, tst_u, &tst_zero, &tst_scale)\
  }\
\
  free(src);\
  free(ref_u);\
  free(tst_u);\
}

#define FUNC_CALLER_QUANT_SYM(pos, size, inbits, outbits, src, tst_s, scale_addr)\
  quantize_symmetric_f##inbits##_s##outbits(src + pos * size, tst_s + pos * size,\
    scale_addr, size, 0, -1);

#define TEST_QUANT_SYM(inbits, outbits) \
static void test_quant_sym_f##inbits##_s##outbits(uint32_t size) {\
  if (size < 4) size = 4;\
  printf("Test symmetrical quantization f"#inbits" -> s"#outbits":\n");\
  printf("num_elements = %u\n", size);\
\
  const uint32_t num_iters = 40000000 / size;\
  if (num_iters <= 2) {\
    printf("Problem size too large.\n");\
    return;\
  }\
\
  int##outbits##_t * const ref_s =\
    (int##outbits##_t *)malloc(size * (outbits >> 3));\
  int##outbits##_t * const tst_s =\
    (int##outbits##_t *)malloc(num_iters * size * (outbits >> 3));\
  float##inbits##_t * const src =\
    (float##inbits##_t *)malloc(num_iters * size * (inbits >> 3));\
\
  const long sint_max = (uint##outbits##_t)-1 >> 1;\
  const long sint_min = (-sint_max) + (-1);\
  srand(time(NULL));\
  for (uint32_t pos = 0; pos < size; ++pos) {\
    ref_s[pos] = (long)rand() % (2 * sint_max + 2) + sint_min;\
  }\
  const uint32_t extreme_pos = rand() % size;\
  ref_s[extreme_pos] = (rand() & 1) ? sint_min : sint_max;\
  const float##inbits##_t ref_scale =\
    (float##inbits##_t)(rand() + 1) / (RAND_MAX >> 2);\
  printf("Generate fp"#inbits" src data with ref_scale = %.2e\n", ref_scale);\
  for (uint32_t pos = 0; pos < size; ++pos) {\
    float##inbits##_t fluc =\
      ((float##inbits##_t)rand() / RAND_MAX - (float##inbits##_t)0.5)\
      * (float##inbits##_t)0.9875;\
    if (pos == extreme_pos) fluc = 0.0;\
    else if (ref_s[pos] == sint_min && fluc < 0) fluc *= -1.0;\
    else if (ref_s[pos] == sint_max && fluc > 0) fluc *= -1.0;\
    src[pos] = ((float##inbits##_t)ref_s[pos] + fluc) * ref_scale;\
  }\
  for (uint32_t pos = 1; pos < num_iters; ++pos) {\
    memcpy(src + pos * size, src, size * (inbits >> 3));\
  }\
  printf("First 4 elements of fp"#inbits" src:\n%.2e, %.2e, %.2e, %.2e\n",\
    src[0], src[1], src[2], src[3]);\
  printf("First 4 elements of s"#outbits" ref_dst:\n%d, %d, %d, %d\n",\
    ref_s[0], ref_s[1], ref_s[2], ref_s[3]);\
\
  float##inbits##_t tst_scale;\
  quantize_symmetric_f##inbits##_s##outbits(\
    src, tst_s, &tst_scale, size, 0, -1);\
\
  printf("relative difference between ref_scale and tst_scale: %.2e\n",\
    (tst_scale - ref_scale) / ref_scale);\
  int eql = 1;\
  for (uint32_t pos = 0; pos < size; ++pos) {\
    if (eql != 0 && tst_s[pos] != ref_s[pos]) {\
      eql = 0;\
      printf("s"#outbits" results at pos %u are inconsistent: ref = %d, tst = %d\n",\
        pos, ref_s[pos], tst_s[pos]);\
      break;\
    }\
  }\
  if (eql != 0) {\
    printf("s"#outbits" results are equal\n");\
    TEST_1D_OPERATION_PERF(size, num_iters, FUNC_CALLER_QUANT_SYM,\
      inbits, outbits, src, tst_s, &tst_scale)\
  }\
\
  free(src);\
  free(ref_s);\
  free(tst_s);\
}

#define FUNC_CALLER_DEQUANT(pos, size, inbits, outbits, src, tst_f, scale) \
  dequantize_symmetric_f##outbits##_s##inbits(src + pos * size,\
    tst_f + pos * size, scale, size);

#define TEST_DEQUANT_SYM(inbits, outbits) \
static void test_dequant_sym_f##outbits##_s##inbits(uint32_t size) {\
  if (size < 4) size = 4;\
  printf("Test dequantization s"#inbits" -> f"#outbits":\n");\
  printf("num_elements = %u\n", size);\
\
  const uint32_t num_iters = 40000000 / size;\
  if (num_iters <= 2) {\
    printf("Problem size too large.\n");\
    return;\
  }\
\
  int##inbits##_t * const src =\
    (int##inbits##_t *)malloc(num_iters * size * (inbits >> 3));\
  float##outbits##_t * const ref_f =\
    (float##outbits##_t *)malloc(size * (outbits >> 3));\
  float##outbits##_t * const tst_f =\
    (float##outbits##_t *)malloc(num_iters * size * (outbits >> 3));\
\
  srand(time(NULL));\
  const float##outbits##_t scale = (float##outbits##_t)rand() / RAND_MAX;\
  printf("Generate src with scale = %.2e\n", scale);\
  for (uint32_t pos = 0; pos < size; ++pos) {\
    src[pos] = (long long)rand() - (long long)(RAND_MAX >> 1);\
    ref_f[pos] = scale * src[pos];\
  }\
  for (uint32_t pos = 1; pos < num_iters; ++pos) {\
    memcpy(src + pos * size, src, size * (inbits >> 3));\
  }\
  printf("First 4 elements of src:\n%d, %d, %d, %d\n",\
    src[0], src[1], src[2], src[3]);\
  printf("First 4 elements of ref:\n%.2e, %.2e, %.2e, %.2e\n",\
    ref_f[0], ref_f[1], ref_f[2], ref_f[3]);\
\
  dequantize_symmetric_f##outbits##_s##inbits(src, tst_f, scale, size);\
\
  float##outbits##_t max_diff = 0.0;\
  for (uint32_t pos = 0; pos < size; ++pos) {\
    float##outbits##_t tmp = tst_f[pos] - ref_f[pos];\
    if (tmp < 0) tmp *= -1.0;\
    if (tmp > max_diff) max_diff = tmp;\
  }\
  printf("Max diff. between tst. and ref.: %.2e\n", max_diff);\
\
  TEST_1D_OPERATION_PERF(size, num_iters, FUNC_CALLER_DEQUANT, inbits, outbits,\
    src, tst_f, scale)\
\
  free(src);\
  free(ref_f);\
  free(tst_f);\
}

#define FUNC_CALLER_REQUANT_UNSYM(pos, size, inbits, fp, outbits,\
  src, dst, org_scale, zero_addr) \
  fp tmp_scale = org_scale;\
  requantize_asymmetric_##inbits##to##outbits(\
    src + pos * size, dst + pos * size, &tmp_scale, zero_addr, size, 0, -1);

#define TEST_REQUANT_UNSYM(fp, inbits, outbits) \
static void test_requant_int##inbits##_t_##fp##_uint##outbits##_t(\
  uint32_t size, int##inbits##_t min_src, int##inbits##_t max_src,\
  fp org_scale) {\
\
  if (max_src < min_src) {\
    int##inbits##_t tmp = min_src;\
    min_src = max_src;\
    max_src = tmp;\
  }\
  if (size < 4) size = 4;\
  printf("Test unsymmetrical requantization int"#inbits"_t -> uint"#outbits"_t:\n");\
  printf("Range of src: %lld - %lld\n", (long long)min_src, (long long)max_src);\
  printf("original_scale = %.2e\n", org_scale);\
  printf("num_elements = %u\n", size);\
\
  const uint32_t num_iters = 40000000 / size;\
  if (num_iters <= 2) {\
    printf("Problem size too large.\n");\
    return;\
  }\
\
  int##inbits##_t * const src = (int##inbits##_t *)malloc(\
    num_iters * size * sizeof(int##inbits##_t));\
  uint##outbits##_t * const dst = (uint##outbits##_t *)malloc(\
    num_iters * size * sizeof(uint##outbits##_t));\
\
  const double range = (long long)max_src - (long long)min_src;\
  srand(time(NULL));\
  if (range == 0) {\
    for (uint32_t pos = 0; pos < size; ++pos) {\
      src[pos] = min_src;\
    }\
  } else {\
    for (uint32_t pos = 0; pos < size; ++pos) {\
      double rv = (double)rand() / (double)RAND_MAX;\
      double v = rv * range + (double)min_src;\
      int##inbits##_t iv = v;\
      if (iv < min_src) iv = min_src;\
      if (iv > max_src) iv = max_src;\
      src[pos] = iv;\
    }\
    uint32_t min_pos = rand() % size;\
    uint32_t max_pos = rand() % size;\
    while(max_pos == min_pos) {\
      max_pos = rand() % size;\
    }\
    src[min_pos] = min_src;\
    src[max_pos] = max_src;\
  }\
  printf("First 4 src elements: %lld, %lld, %lld, %lld\n",\
    (long long)src[0], (long long)src[1], (long long)src[2], (long long)src[3]);\
  for (uint32_t it = 1; it < num_iters; ++it) {\
    memcpy(src + it * size, src, size * sizeof(int##inbits##_t));\
  }\
  for (uint32_t pos = 0; pos < size; ++pos) {\
    dst[pos] = rand();\
  }\
\
  const long long renorm_min_src = min_src > 0 ? 0 : min_src;\
  const long long renorm_max_src = max_src < 0 ? 0 : max_src;\
  const fp ref_scale = (double)org_scale * \
    (double)(renorm_max_src - renorm_min_src) / ((uint##outbits##_t)-1);\
  printf("ref_scale = %.2e\n", ref_scale);\
\
  uint##outbits##_t zero_point;\
  fp new_scale = org_scale;\
  requantize_asymmetric_##inbits##to##outbits(src, dst,\
  &new_scale, &zero_point, size, 0, -1);\
\
  printf("tst_zero = %u\n", zero_point);\
  printf("tst_scale - ref_scale = %.2e\n", new_scale - ref_scale);\
  long min_out, max_out;\
  double max_diff_out = 0.0;\
  min_out = max_out = dst[0];\
  for (uint32_t pos = 0; pos < size; ++pos) {\
    long ld = dst[pos];\
    if (ld < min_out) min_out = ld;\
    if (ld > max_out) max_out = ld;\
    double curr_fp = src[pos] * (double)org_scale;\
    double curr_i8 = curr_fp / (double)new_scale;\
    double curr_u8 = curr_i8 + (double)zero_point;\
    double tmp_diff_out = (double)ld - curr_u8;\
    if (tmp_diff_out < 0) tmp_diff_out *= -1.0;\
    if (tmp_diff_out > max_diff_out) max_diff_out = tmp_diff_out;\
  }\
  printf("range of requant u"#outbits": [%ld, %ld]\n", min_out, max_out);\
  printf("max deviation of requant u"#outbits": %.2e\n", max_diff_out);\
\
  TEST_1D_OPERATION_PERF(size, num_iters, FUNC_CALLER_REQUANT_UNSYM,\
    inbits, fp, outbits, src, dst, org_scale, &zero_point)\
\
  free(src);\
  free(dst);\
}

#define FUNC_CALLER_REQUANT_SYM(pos, size, inbits, fp, outbits, src, dst, org_scale) \
  fp tmp_scale = org_scale;\
  requantize_symmetric_##inbits##to##outbits(\
    src + pos * size, dst + pos * size, &tmp_scale, size, 0, -1);

#define TEST_REQUANT_SYM(fp, inbits, outbits) \
static void test_requant_int##inbits##_t_##fp##_int##outbits##_t(\
  uint32_t size, int##inbits##_t max_abs, fp org_scale) {\
\
  if (max_abs < 0) max_abs = -max_abs;\
  if (size < 4) size = 4;\
  printf("Test symmetrical requantization int"#inbits"_t -> int"#outbits"_t:\n");\
  printf("Range of src: %d - %d\n", -max_abs, max_abs);\
  printf("original_scale = %.2e\n", org_scale);\
  printf("num_elements = %u\n", size);\
\
  const uint32_t num_iters = 40000000 / size;\
  if (num_iters <= 2) {\
    printf("Problem size too large.\n");\
    return;\
  }\
\
  int##inbits##_t * const src = (int##inbits##_t *)malloc(\
    num_iters * size * sizeof(int##inbits##_t));\
  int##outbits##_t * const dst = (int##outbits##_t *)malloc(\
    num_iters * size * sizeof(int##outbits##_t));\
\
  srand(time(NULL));\
  if (max_abs == 0) {\
    memset(src, 0, size * sizeof(int##inbits##_t));\
  } else {\
    const double rand_range = 2.0 * (double)max_abs + 1.0;\
    const double rand_offset = -1.0 * (double)max_abs;\
    for (uint32_t pos = 0; pos < size; ++pos) {\
      double rv = (double)rand() / (double)RAND_MAX;\
      double ra = rv * rand_range + rand_offset;\
      int##inbits##_t ia = ra;\
      if (ia < -max_abs) ia = -max_abs;\
      if (ia > max_abs) ia = max_abs;\
      src[pos] = ia;\
    }\
    uint32_t max_rand_pos = rand() % size;\
    src[max_rand_pos] = (rand() & 1) ? max_abs : -max_abs;\
  }\
  printf("The first 4 elements of src: %lld, %lld, %lld, %lld\n",\
    (long long)src[0], (long long)src[1], (long long)src[2], (long long)src[3]);\
  for (uint32_t it = 1; it < num_iters; ++it) {\
    memcpy(src + it * size, src, size * sizeof(int##inbits##_t));\
  }\
\
  const fp ref_scale = (double)org_scale * (double)max_abs / \
    (double)(((uint##outbits##_t)-1) >> 1);\
  printf("ref_scale = %.2e\n", ref_scale);\
  fp new_scale = org_scale;\
  requantize_symmetric_##inbits##to##outbits(src, dst, &new_scale, size, 0, -1);\
  printf("diff. between ref_scale and tst_scale: %.2e\n",\
    new_scale - ref_scale);\
\
  int##outbits##_t max_out_abs = 0;\
  double max_out_dev = 0.0;\
  for (uint32_t pos = 0; pos < size; ++pos) {\
    int##outbits##_t l1 = dst[pos];\
    if (l1 > max_out_abs) max_out_abs = l1;\
    if (-l1 > max_out_abs) max_out_abs = -l1;\
    if (new_scale != 0.0) {\
      double expected = (double)src[pos] * (double)org_scale / \
        (double)new_scale;\
      double tmp_dev = expected - (double)dst[pos];\
      if (tmp_dev < 0) tmp_dev *= -1.0;\
      if (tmp_dev > max_out_dev) max_out_dev = tmp_dev;\
    }\
  }\
  printf("max abs of output int"#outbits": %d\n", max_out_abs);\
  if (new_scale == 0.0) {\
    printf("max deviation of output int"#outbits" not determined.\n");\
  } else {\
    printf("max deviation of output int"#outbits": %.2e\n", max_out_dev);\
  }\
\
  TEST_1D_OPERATION_PERF(size, num_iters, FUNC_CALLER_REQUANT_SYM,\
    inbits, fp, outbits, src, dst, org_scale)\
\
  free(src);\
  free(dst);\
}

#endif
