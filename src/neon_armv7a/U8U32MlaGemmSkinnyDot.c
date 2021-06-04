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


#ifndef GEMM_UNSIGNED_INT
#define GEMM_UNSIGNED_INT
#endif

#include "arm_neon/ARMCompareAndSwap.h"
#include "arm_neon/NeonI8I32MlaGemmSkinnyDot.h"
#include "common/CommonSkinnyDot.h"

GEMM_SKINNY_DOT_PARALLEL_FUNC(u8u32mlagemm, 1, 15, 7, 131072, uint8_t, uint8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(u8u32mlagemm, 2, 15, 7, 131072, uint8_t, uint8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(u8u32mlagemm, 3, 15, 3, 131072, uint8_t, uint8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(u8u32mlagemm, 4, 15, 3, 131072, uint8_t, uint8_t)