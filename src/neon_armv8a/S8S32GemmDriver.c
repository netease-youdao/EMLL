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


#include "neon_armv8a/S8S32MlaGemmDriver.h"
#include "neon_armv8a/S8S32DotGemmDriver.h"
#include "arm_neon/ARMCpuType.h"

int s8s32gemm_serial(int a_rowmajor, int b_rowmajor,
  const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t N, uint32_t K, int32_t beta_inp) {

  if (blas_arm_get_i8i32_support() == 2) {
    return s8s32dotgemm_serial(a_rowmajor, b_rowmajor, A, B, C,
      M, N, K, beta_inp);
  } else {
    return s8s32mlagemm_serial(a_rowmajor, b_rowmajor, A, B, C,
      M, N, K, beta_inp);
  }
}

int s8s32gemm(int a_rowmajor, int b_rowmajor,
  const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t N, uint32_t K,
  int32_t beta_inp, uint32_t num_threads) {

  if (blas_arm_get_i8i32_support() == 2) {
    return s8s32dotgemm(a_rowmajor, b_rowmajor, A, B, C,
      M, N, K, beta_inp, num_threads);
  } else {
    return s8s32mlagemm(a_rowmajor, b_rowmajor, A, B, C,
      M, N, K, beta_inp, num_threads);
  }
}

