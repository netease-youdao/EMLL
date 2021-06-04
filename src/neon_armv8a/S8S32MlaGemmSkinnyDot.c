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


#ifdef GEMM_UNSIGNED_INT
#undef GEMM_UNSIGNED_INT
#endif

#include "arm_neon/ARMCompareAndSwap.h"
#include "neon_armv8a/I8I32MlaGemmSkinnyDot.h"
#include "common/CommonSkinnyDot.h"

GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(s8s32mlagemm, 1, 31, 5, 131072, int8_t, int8_t, unroll_test)
GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(s8s32mlagemm, 2, 31, 5, 131072, int8_t, int8_t, unroll_test)
GEMM_SKINNY_DOT_PARALLEL_FUNC_NOINCINLINE(s8s32mlagemm, 3, 31, 5, 131072, int8_t, int8_t, unroll_test)

GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32mlagemm, 4, 15, 7, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32mlagemm, 5, 15, 7, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32mlagemm, 6, 15, 7, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32mlagemm, 7, 15, 3, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32mlagemm, 8, 15, 3, 131072, int8_t, int8_t)
