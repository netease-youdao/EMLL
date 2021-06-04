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

void hgemm_float16_t_float16_t_ncopy_unroll8(const float16_t * __restrict__ src,
  float16_t * __restrict__ dst, uint32_t ld_dim, uint32_t dim1, uint32_t dim2);

void hgemm_float16_t_float16_t_ncopy_unroll16(const float16_t * __restrict__ src,
  float16_t * __restrict__ dst, uint32_t ld_dim, uint32_t dim1, uint32_t dim2);

void hgemm_float16_t_float16_t_tcopy_unroll8(const float16_t * __restrict__ src,
  float16_t * __restrict__ dst, uint32_t ld_dim, uint32_t dim1, uint32_t dim2);

void hgemm_float16_t_float16_t_tcopy_unroll16(const float16_t * __restrict__ src,
  float16_t * __restrict__ dst, uint32_t ld_dim, uint32_t dim1, uint32_t dim2);

