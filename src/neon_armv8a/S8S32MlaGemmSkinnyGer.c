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
#include "arm_neon/NeonI8I32MlaGemmSkinnyGer.h"

GEMM_SKINNY_GER_PARALLEL_FUNC(s8s32mlagemm, 1, 5, 29, 8192, int8_t, int8_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(s8s32mlagemm, 2, 5, 29, 8192, int8_t, int8_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(s8s32mlagemm, 3, 5, 29, 8192, int8_t, int8_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(s8s32mlagemm, 4, 5, 29, 8192, int8_t, int8_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(s8s32mlagemm, 5, 5, 13, 8192, int8_t, int8_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(s8s32mlagemm, 6, 5, 13, 8192, int8_t, int8_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(s8s32mlagemm, 7, 5, 13, 8192, int8_t, int8_t)
GEMM_SKINNY_GER_PARALLEL_FUNC(s8s32mlagemm, 8, 5, 13, 8192, int8_t, int8_t)
