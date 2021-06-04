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
#include <arm_neon.h>

int hgemm_serial(uint8_t transAB, const float16_t *A, const float16_t *B, float16_t *C,
  uint32_t M, uint32_t N, uint32_t K, float16_t beta_inp);

int hgemm(uint8_t transAB, const float16_t *A, const float16_t *B, float16_t *C,
  uint32_t M, uint32_t N, uint32_t K, float16_t beta_inp, uint32_t num_threads);
