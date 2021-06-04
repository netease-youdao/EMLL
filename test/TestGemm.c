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


#include "common/CommonTest.h"
#include "Gemm.h"

STD_TEST(sgemm, float, float, float, (RAND_MAX >> 2), (RAND_MAX >> 2))

STD_TEST(s8s32gemm, int8_t, int8_t, int32_t, 64, 1)

STD_TEST(u8u32gemm, uint8_t, uint8_t, uint32_t, -64, 1)

#if __aarch64__
STD_TEST(hgemm, float16_t, float16_t, float16_t, 6, 32)
#endif

/*************************************************************************
 * cmd usage of test program for GEMM functions
 *   <path/to/test_emll_gemm> <M> <N> <K> <transAB> <num_threads>\
 *      <gemm_type> <beta>
 * GEMM operation: C[MxN] = A[MxK] B[KxN] + beta * C[MxN]
 * Parameters:
 *   M: the number of rows in matrix A.
 *   N: the number of columns in matrix B.
 *   K: the number of columns in matrix A.
 *   transAB: a number indicating the storage order of source matrices:
 *     0: A column-major, B column-major
 *     1: A row-major, B column-major
 *     2: A column-major, B row-major
 *     3: A row-major, B row-major
 *   num_threads: number of threads for GEMM.
 *   gemm_type: a string indicating the type of GEMM:
 *     sgemm: fp32 GEMM
 *     hgemm: fp16 GEMM
 *     u8u32: uint8 * uint8 -> uint32 GEMM
 *     s8s32: int8 * int8 -> int32 GEMM
 *   beta: the scaling factor applied to matrix C prior to GEMM operation,
 *         C = AB + beta * C.
 *************************************************************************/
int main(int argc, char **argv) {
  uint32_t M = 383;
  uint32_t N = 479;
  uint32_t K = 319;
  uint8_t transAB = 0;
  uint32_t num_threads = 0;
  const char *gemm_type = "sgemm";
  double beta = 0.5;
  if (argc > 1) M = atoi(argv[1]);
  if (argc > 2) N = atoi(argv[2]);
  if (argc > 3) K = atoi(argv[3]);
  if (argc > 4) transAB = atoi(argv[4]);
  if (argc > 5) num_threads = atoi(argv[5]);
  if (argc > 6) gemm_type = argv[6];
  if (argc > 7) beta = atof(argv[7]);
  printf("Test info: M = %u, N = %u, K = %u\n", M, N, K);
  printf("Test info: a_rowmajor = %d, b_rowmajor = %d\n",
    transAB & 1, (transAB & 2) >> 1);
  printf("Test info: num_threads = %u, beta = %.2e\n", num_threads, beta);

#if __aarch64__
  if (strcmp(gemm_type, "hgemm") == 0) {
    printf("Test info: gemmtype = hgemm.\n");
    std_test_hgemm(hgemm, M, N, K, transAB, beta, num_threads);
    return 0;
  }
#endif

  if (strcmp(gemm_type, "u8u32") == 0) {
    printf("Test info: gemmtype = u8u32gemm.\n");
    std_test_u8u32gemm(u8u32gemm, M, N, K, transAB, beta, num_threads);
    return 0;
  }

  if (strcmp(gemm_type, "s8s32") == 0) {
    printf("Test info: gemmtype = s8s32gemm.\n");
    std_test_s8s32gemm(s8s32gemm, M, N, K, transAB, beta, num_threads);
    return 0;
  }

  printf("Test info: gemmtype = sgemm.\n");
  std_test_sgemm(sgemm, M, N, K, transAB, beta, num_threads);
  return 0;
}

