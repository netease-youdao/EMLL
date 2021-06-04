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

int u8u32dotgemm_serial(int a_rowmajor, int b_rowmajor,
  const uint8_t *A, const uint8_t *B,
  uint32_t *C, uint32_t M, uint32_t N, uint32_t K, uint32_t beta_inp);

int u8u32dotgemm(int a_rowmajor, int b_rowmajor,
  const uint8_t *A, const uint8_t *B,
  uint32_t *C, uint32_t M, uint32_t N, uint32_t K,
  uint32_t beta_inp, uint32_t num_threads);

