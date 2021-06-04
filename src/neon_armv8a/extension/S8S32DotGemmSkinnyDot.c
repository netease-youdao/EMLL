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
#include "common/CommonSkinnyDot.h"
#include "arm_neon/NeonI8I32DotGemmSkinnyDot.h"

GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32dotgemm, 1, 29, 7, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32dotgemm, 2, 29, 7, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32dotgemm, 3, 29, 7, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32dotgemm, 4, 29, 7, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32dotgemm, 5, 29, 7, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32dotgemm, 6, 29, 7, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32dotgemm, 7, 29, 3, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32dotgemm, 8, 29, 3, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32dotgemm, 9, 29, 3, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32dotgemm, 10, 29, 3, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32dotgemm, 11, 29, 3, 131072, int8_t, int8_t)
GEMM_SKINNY_DOT_PARALLEL_FUNC(s8s32dotgemm, 12, 29, 3, 131072, int8_t, int8_t)
