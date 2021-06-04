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


#include "neon_armv7a/U8U32MlaGemmDriver.h"
#include "arm_neon/ARMCpuType.h"

int u8u32gemm_serial(int a_rowmajor, int b_rowmajor,
  const uint8_t *A, const uint8_t *B,
  uint32_t *C, uint32_t M, uint32_t N, uint32_t K, uint32_t beta_inp) {

  if (blas_arm_get_i8i32_support() == 0) {
    return 2;
  }
  return u8u32mlagemm_serial(a_rowmajor, b_rowmajor, A, B, C,
    M, N, K, beta_inp);
}

int u8u32gemm(int a_rowmajor, int b_rowmajor,
  const uint8_t *A, const uint8_t *B,
  uint32_t *C, uint32_t M, uint32_t N, uint32_t K,
  uint32_t beta_inp, uint32_t num_threads) {

  if (blas_arm_get_i8i32_support() == 0) {
    return 2;
  }
  return u8u32mlagemm(a_rowmajor, b_rowmajor, A, B, C,
    M, N, K, beta_inp, num_threads);
}
