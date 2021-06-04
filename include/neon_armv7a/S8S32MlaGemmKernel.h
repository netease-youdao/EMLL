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


#include <stdint.h>

void s8s32mlagemm_kernel_lm_m6n8(uint32_t M, uint32_t N, uint32_t K,
  int32_t beta,
  const int16_t * __restrict__ sa, const int16_t * __restrict__ sb,
  int32_t * __restrict__ C, uint32_t ldc);

void s8s32mlagemm_kernel_ln_m8n6(uint32_t M, uint32_t N, uint32_t K,
  int32_t beta,
  const int16_t * __restrict__ sa, const int16_t * __restrict__ sb,
  int32_t * __restrict__ C, uint32_t ldc);

