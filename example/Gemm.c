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
 * File:        example/Gemm.c
 * Description: This file is an example of using EMLL in your application.
 *              In this example we do 3 types of fp32 GEMM:
 *              (1) direct SGEMM
 *              (2) asymmetrically quantize to uint8, do GEMM to int32,
 *                  finally dequantize to fp32.
 *              (3) symmetrically quantize to int8, do GEMM to int32,
 *                  finally dequantize to fp32.
 *              Users should tell compilers to include the "include"
 *              directory of the library and link to the static
 *              library of EMLL.
 *****************************************************************************/

#include "Gemm.h"
#include "Quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>

int main(int argc, char **argv) {

  if (argc == 1 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
    printf("Usage: %s [M] [N] [K]\n", argv[0]);
    return 0;
  }

  uint16_t M = 300, N = 400, K = 500;
  if (argc > 1) M = atoi(argv[1]);
  if (argc > 2) N = atoi(argv[2]);
  if (argc > 3) K = atoi(argv[3]);

  if (!M || !N || !K) {
    fprintf(stderr, "Invalid (zero or negative) M, N or K.\n");
    return -1;
  }

  printf("Test matmul C=AB with fp32, symmetric & asymmetric quantizations.\n");
  printf("matrix A (column-major): %u x %u\n", M, K);
  printf("matrix B (column-major): %u x %u\n", K, N);
  printf("matrix C (column-major): %u x %u\n", M, N);
  const uint32_t size_a = (uint32_t)M * (uint32_t)K;
  const uint32_t size_b = (uint32_t)N * (uint32_t)K;
  const uint32_t size_c = (uint32_t)M * (uint32_t)N;

  /* allocate fp32 matrices */
  float * const A_f = (float *)malloc(size_a * 4);
  float * const B_f = (float *)malloc(size_b * 4);
  float * const C_f = (float *)malloc(size_c * 4);

  /* allocate quant-u8 matrices and arrays */
  uint8_t * const A_u = (uint8_t *)malloc(size_a);
  uint8_t * const B_u = (uint8_t *)malloc(size_b);
  int32_t * const C_qu = (int32_t *)malloc(size_c * 4);
  float * const C_fqu = (float *)malloc(size_c * 4);
  uint32_t * const A_sum = (uint32_t *)malloc(M * 4);
  uint32_t * const B_sum = (uint32_t *)malloc(N * 4);

  /* allocate quant-s8 matrices and arrays */
  int8_t * const A_s = (int8_t *)malloc(size_a);
  int8_t * const B_s = (int8_t *)malloc(size_b);
  int32_t * const C_qs = (int32_t *)malloc(size_c * 4);
  float * const C_fqs = (float *)malloc(size_c * 4);

  int ret_status = 0;
  do {
    if (!A_f || !B_f || !C_f || !A_u || !B_u || !C_qu || !C_fqu ||
      !A_sum || !B_sum || !A_s || !B_s || !C_qs || !C_fqs) {
      fprintf(stderr, "Memory allocation failed.\n");
      ret_status = -1;
      break;
    }

    /* prepare data */
    srand(time(NULL));
    for (uint32_t i = 0; i < size_a; ++i) {
      A_f[i] = (float)rand() / (float)RAND_MAX - 0.3;
    }
    for (uint32_t i = 0; i < size_b; ++i) {
      B_f[i] = (float)rand() / (float)RAND_MAX - 0.3;
    }
    printf("Matrix preparation done. rand [-0.3, 0.7)\n");

    /* all matrices are column-major */
    /* example 1: do normal fp32 GEMM */
    /* gemm(a_rowmajor, b_rowmajor, a_addr, b_addr, c_addr, m, n, k, beta, threads) */
    int sgemm_status = sgemm(0, 0, A_f, B_f, C_f, M, N, K, 1, 0);
    if (sgemm_status != 0) {
      fprintf(stderr, "sgemm returns error code %d\n", sgemm_status);
      ret_status = -1;
      break;
    }
    printf("Normal SGEMM done.\n");

    /* example 2: do asymmetric quant 8-bit GEMM */
    float scale_a, scale_b;
    uint8_t zero_point_a, zero_point_b;
    /* quantitize the source matrices */
    /* quant_asym(input_addr, output_addr, &zero_point, &scale, array_length, input_min, input_Max) */
    quantize_asymmetric_f32_u8(A_f, A_u, &zero_point_a, &scale_a, size_a, 0, -1);
    quantize_asymmetric_f32_u8(B_f, B_u, &zero_point_b, &scale_b, size_b, 0, -1);
    /* do unsigned 8->32 bit GEMM */
    /* gemm(a_rowmajor, b_rowmajor, a_addr, b_addr, c_addr, m, n, k, beta, threads) */
    int u8u32_status = u8u32gemm(0, 0, A_u, B_u, (uint32_t *)C_qu, M, N, K, 1, 0);
    if (u8u32_status != 0) {
      fprintf(stderr, "u8u32gemm returns error code %d\n", u8u32_status);
      ret_status = -1;
      break;
    }
    /* sum row/col of source matrices (along K dim) */
    u8u32_sum(A_u, A_sum, M, K, 0);
    u8u32_sum(B_u, B_sum, K, N, 1);
    /* bias the result of 8->32 bit GEMM */
    bias_int32_t(C_qu,
      (int32_t)zero_point_a * (int32_t)zero_point_b * (int32_t)K,
      (int32_t *)A_sum, -(int32_t)zero_point_b,
      (int32_t *)B_sum, -(int32_t)zero_point_a, M, N);
    /* dequantitize the result */
    /* dequant(input_addr, output_addr, scale, array_length) */
    dequantize_symmetric_f32_s32(C_qu, C_fqu, scale_a * scale_b, size_c);
    printf("Asym quant GEMM done.\n");

    /* example 3: do symmetric quant 8-bit GEMM */
    /* quantitize the source matrices */
    /* quant_sym(input_addr, output_addr, &scale, array_length, input_min, input_Max) */
    quantize_symmetric_f32_s8(A_f, A_s, &scale_a, size_a, 0, -1);
    quantize_symmetric_f32_s8(B_f, B_s, &scale_b, size_b, 0, -1);
    /* do signed 8->32 bit GEMM */
    int s8s32_status = s8s32gemm(0, 0, A_s, B_s, C_qs, M, N, K, 1, 0);
    if (s8s32_status != 0) {
      fprintf(stderr, "s8s32gemm returns error code %d\n", s8s32_status);
      ret_status = -1;
      break;
    }
    /* dequantitize the result */
    /* dequant(input_addr, output_addr, scale, array_length) */
    dequantize_symmetric_f32_s32(C_qs, C_fqs, scale_a * scale_b, size_c);
    printf("Sym quant GEMM done.\n");

    /* evaluate the results */
    float max_diff_qu = 0, max_diff_qs = 0;
    double sum_diff_sqr_qu = 0, sum_diff_sqr_qs = 0;
    for (uint32_t i = 0; i < size_c; ++i) {
      float tmp_diff_qu = fabsf(C_fqu[i] - C_f[i]);
      float tmp_diff_qs = fabsf(C_fqs[i] - C_f[i]);
      max_diff_qu = fmaxf(max_diff_qu, tmp_diff_qu);
      max_diff_qs = fmaxf(max_diff_qs, tmp_diff_qs);
      sum_diff_sqr_qu += max_diff_qu * max_diff_qu;
      sum_diff_sqr_qs += max_diff_qs * max_diff_qs;
    }
    double std_dev_qu = size_c == 1 ? 0 : sqrt(sum_diff_sqr_qu / (size_c - 1));
    double std_dev_qs = size_c == 1 ? 0 : sqrt(sum_diff_sqr_qs / (size_c - 1));
    printf("The results of asym quant compared to std fp32: ");
    printf("max_diff = %.2e, stdev = %.2e\n", max_diff_qu, std_dev_qu);
    printf("The results of sym quant compared to std fp32: ");
    printf("max_diff = %.2e, stdev = %.2e\n", max_diff_qs, std_dev_qs);
  } while (false);

  /* clean up */
  free(A_f);
  free(B_f);
  free(C_f);
  free(A_u);
  free(B_u);
  free(C_qu);
  free(C_fqu);
  free(A_sum);
  free(B_sum);
  free(A_s);
  free(B_s);
  free(C_qs);
  free(C_fqs);
  return ret_status;
}
