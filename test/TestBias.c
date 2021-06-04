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


#if __aarch64__
#include "neon_armv8a/Bias.h"
#else
#include "neon_armv7a/Bias.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>

#define TEST_BIAS(type) \
static void test_bias_##type(uint32_t dim1, uint32_t dim2, uint8_t status) {\
  bool dim0_bias = status & 1;\
  bool dim1_bias = status & 2;\
  bool dim2_bias = status & 4;\
  printf("Test info for bias:\n");\
  printf("data type = "#type"\n");\
  printf("dim1 = %u, dim2 = %u\n", dim1, dim2);\
  printf("dim0_bias = %d\n", dim0_bias ? 1 : 0);\
  printf("dim1_bias = %d\n", dim1_bias ? 1 : 0);\
  printf("dim2_bias = %d\n", dim2_bias ? 1 : 0);\
\
  const uint64_t size_dat = (dim1 + 4) * (dim2 + 4);\
  const uint32_t num_iters = 40000000 / size_dat;\
  if (num_iters <= 2) {\
    printf("Problem size too large.\n");\
    return;\
  }\
  type * const ref = (type *)malloc(sizeof(type) * size_dat);\
  type * const dat = (type *)malloc(sizeof(type) *\
    size_dat * num_iters);\
  type * const bias_dim1 = dim1_bias ? (type *)malloc(sizeof(type) *\
    (dim1 + 4) * num_iters) : NULL;\
  type * const bias_dim2 = dim2_bias ? (type *)malloc(sizeof(type) *\
    (dim2 + 4) * num_iters) : NULL;\
\
  srand(time(NULL));\
  for (uint64_t pos = 0; pos < size_dat; ++pos) {\
    ref[pos] = rand() % 256;\
  }\
  for (uint32_t pos = 0; pos < num_iters; ++pos) {\
    memcpy(dat + pos * size_dat, ref, size_dat * sizeof(type));\
  }\
  if (dim1_bias) {\
    for (uint32_t pos = 0; pos < dim1 + 4; ++pos) {\
      bias_dim1[pos] = rand() % 256;\
    }\
    for (uint32_t pos = 1; pos < num_iters; ++pos) {\
      memcpy(bias_dim1 + pos * (dim1 + 4), bias_dim1, (dim1 + 4) * sizeof(type));\
    }\
  }\
  if (dim2_bias) {\
    for (uint32_t pos = 0; pos < dim2 + 4; ++pos) {\
      bias_dim2[pos] = rand() % 256;\
    }\
    for (uint32_t pos = 1; pos < num_iters; ++pos) {\
      memcpy(bias_dim2 + pos * (dim2 + 4), bias_dim2, (dim2 + 4) * sizeof(type));\
    }\
  }\
\
  const type bias_v0 = dim0_bias ? (rand() % 256 + 1) : 0;\
  if (dim0_bias) {\
    for (uint32_t pos = 0; pos < dim1 * dim2; ++pos) ref[pos] += bias_v0;\
  }\
\
  if (dim1_bias) {\
    for (uint32_t dim2_pos = 0; dim2_pos < dim2; ++dim2_pos) {\
      type *curr = ref + dim2_pos * dim1;\
      for (uint32_t dim1_pos = 0; dim1_pos < dim1; ++dim1_pos) {\
        curr[dim1_pos] += (type)2.0 * bias_dim1[dim1_pos];\
      }\
    }\
  }\
\
  if (dim2_bias) {\
    for (uint32_t dim2_pos = 0; dim2_pos < dim2; ++dim2_pos) {\
      const type bias = (type)3.0 * bias_dim2[dim2_pos];\
      type *curr = ref + dim2_pos * dim1;\
      for (uint32_t dim1_pos = 0; dim1_pos < dim1; ++dim1_pos) {\
        curr[dim1_pos] += bias;\
      }\
    }\
  }\
\
  bias_##type(dat, bias_v0, bias_dim1, 2.0, bias_dim2, 3.0, dim1, dim2);\
  double max_diff = 0.0;\
  for (uint64_t pos = 0; pos < size_dat; ++pos) {\
    double tmp = (double)dat[pos] - (double)ref[pos];\
    if (tmp < 0) tmp *= -1.0;\
    if (tmp > max_diff) max_diff = tmp;\
  }\
  printf("Max diff. between calc. and ref.: %.2e\n", max_diff);\
\
  struct timespec st, et;\
  clock_gettime(CLOCK_MONOTONIC, &st);\
  for (uint32_t pos = 1; pos < num_iters; ++pos) {\
    bias_##type(dat + pos * size_dat, bias_v0,\
      bias_dim1 ? bias_dim1 + pos * dim1 : NULL, 2.0,\
      bias_dim2 ? bias_dim2 + pos * dim2 : NULL, 3.0,\
      dim1, dim2);\
  }\
  clock_gettime(CLOCK_MONOTONIC, &et);\
  double nsec = (double)(et.tv_nsec - st.tv_nsec) + 1.0e9 * (double)\
    (et.tv_sec - st.tv_sec);\
  printf("Avg. perf.: %.2e G elements per second.\n", (double)dim1 * \
    (double)dim2 * (double)(num_iters - 1) / nsec);\
\
  free(ref);\
  free(dat);\
  free(bias_dim1);\
  free(bias_dim2);\
}

#define TEST_SUM(signint, sumfunc) \
void test_sum_##signint##8to32(uint32_t dim1, uint32_t dim2,\
  uint32_t status) {\
\
  printf("Test info for sum:\n");\
  printf("data type = "#signint"8 -> "#signint"32\n");\
  printf("dim1 = %u, dim2 = %u\n", dim1, dim2);\
  if (status) {\
    printf("sum along dim1 direction, output length = dim2\n");\
  } else {\
    printf("sum along dim2 direction, output length = dim1\n");\
  }\
\
  const uint64_t size_dat = (dim1 + 4) * (dim2 + 4);\
  const uint32_t num_iters = 40000000 / size_dat;\
  if (num_iters <= 2) {\
    printf("Problem size too large.\n");\
    return;\
  }\
  signint##8_t * const dat = (signint##8_t *)malloc(size_dat * num_iters);\
\
  const uint32_t size_out = status ? (dim2 + 4) : (dim1 + 4);\
  signint##32_t * const ref = (signint##32_t *)malloc(size_out * 4);\
  signint##32_t * const tst = (signint##32_t *)malloc(size_out * num_iters * 4);\
\
  srand(time(NULL));\
  for (uint64_t pos = 0; pos < size_dat; ++pos) {\
    dat[pos] = rand();\
  }\
  for (uint32_t pos = 1; pos < num_iters; ++pos) {\
    memcpy(dat + pos * size_dat, dat, size_dat);\
  }\
\
  if (status) {\
    for (uint32_t dim2_pos = 0; dim2_pos < dim2; ++dim2_pos) {\
      const signint##8_t *src = dat + dim2_pos * dim1;\
      signint##32_t sum = 0;\
      for (uint32_t dim1_pos = 0; dim1_pos < dim1; ++dim1_pos) {\
        sum += src[dim1_pos];\
      }\
      ref[dim2_pos] = sum;\
    }\
    for (uint32_t dim2_pos = dim2; dim2_pos < size_out; ++dim2_pos) {\
      ref[dim2_pos] = tst[dim2_pos] = rand();\
    }\
  } else {\
    for (uint32_t dim1_pos = 0; dim1_pos < dim1; ++dim1_pos) {\
      ref[dim1_pos] = 0;\
    }\
    for (uint32_t dim1_pos = dim1; dim1_pos < size_out; dim1_pos++) {\
      ref[dim1_pos] = tst[dim1_pos] = rand();\
    }\
    for (uint32_t dim2_pos = 0; dim2_pos < dim2; ++dim2_pos) {\
      const signint##8_t *src = dat + dim2_pos * dim1;\
      for (uint32_t dim1_pos = 0; dim1_pos < dim1; ++dim1_pos) {\
        ref[dim1_pos] += src[dim1_pos];\
      }\
    }\
  }\
\
  sumfunc(dat, tst, dim1, dim2, status);\
  int consistent = 1;\
  for (uint32_t pos = 0; pos < size_out; ++pos) {\
    if (consistent != 0 && ref[pos] != tst[pos]) {\
      consistent = 0;\
      printf("elements at pos %u are unequal.\n", pos);\
    }\
  }\
  if (consistent != 0) {\
    printf("all elements are equal between ref. and tst.\n");\
    struct timespec st, et;\
    clock_gettime(CLOCK_MONOTONIC, &st);\
    for (uint32_t pos = 1; pos < num_iters; ++pos) {\
      sumfunc(dat + pos * size_dat, tst + pos * size_out,\
        dim1, dim2, status);\
    }\
    clock_gettime(CLOCK_MONOTONIC, &et);\
    double nsec = (double)(et.tv_nsec - st.tv_nsec) + 1.0e9 * \
      (double)(et.tv_sec - st.tv_sec);\
    printf("Avg. Perf.: %.2e G elements read per second.\n",\
      (double)dim1 * (double)(dim2) * (double)(num_iters - 1) / nsec);\
  }\
  free(dat);\
  free(ref);\
  free(tst);\
}


TEST_BIAS(float)

TEST_BIAS(int32_t)

TEST_SUM(uint, u8u32_sum)

/************************************************************************
 * cmd usage of the test program for bias functions
 *   <path/to/test_emll_bias> <dim1> <dim2> <bias_status> <data_type>
 * dim1: the length of the first dimension of the matrix,
 *       equals to number of columns for row-major matrices,
 *       equals to number of rows for column-major matrices.
 * dim2: the length of the second dimension of the matrix,
 *       equals to number of rows for row-major matrices,
 *       equals to number of columns for column-major matrices.
 * bias_status: a number indicating which function to test.
 *       0 - 7 for bias function:
 *         0: no bias is performed.
 *         1: only scalar bias is applied. the bias is identical
 *            to each element.
 *         2: bias only along the first dimension, the size of bias
 *            vector equals dim1. for row-major matrix, this means
 *            elem(col_id, row_id) += bias(col_id);
 *         3: scalar & first-dimension bias operations.
 *         4: bias only along the second dimension, the size of bias
 *            vector equals dim2. for row-major matrix, this means
 *            elem(col_id, row_id) += bias(row_id);
 *         5: scalar & second-dimension bias operations.
 *         6: first-dimension & second-dimension bias operations.
 *         7: scalar & first-dimension & second-dimension bias operations.
 *       8 - 9 for sum function:
 *         8: sum along dim2. for a row-major matrix, the sum of elements
 *            in each column is calculated.
 *         9: sum along dim1. for a row-major matrix, the sum of elements
 *            in each row is calculated.
 * data_type: a string indicating the data type of bias
 *         float: fp32
 *         int32: int32_t
 ************************************************************************/

int main(int argc, char **argv) {

  const uint32_t dim1 = (argc > 1) ? atoi(argv[1]) : 63;
  const uint32_t dim2 = (argc > 2) ? atoi(argv[2]) : 143;
  const uint8_t bias_status = (argc > 3) ? atoi(argv[3]) : 7;
  const char * const data_type = (argc > 4) ? argv[4] : "float";

  if (bias_status > 7) {
    test_sum_uint8to32(dim1, dim2, bias_status & 1);
    return 0;
  }

  if (data_type[0] == 'f' || data_type[0] == 'F') {
    test_bias_float(dim1, dim2, bias_status);
    return 0;
  }

  test_bias_int32_t(dim1, dim2, bias_status);
  return 0;
}

