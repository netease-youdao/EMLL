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


#include "neon_armv8a/S8S32DotGemmCopy.h"
#include "neon_armv8a/S8S32DotGemmKernel.h"
#include "neon_armv8a/S8S32DotGemmSkinnyDot.h"
#include "arm_neon/ARMCompareAndSwap.h"
#include "common/CommonDriver.h"

#ifdef SCRATCH_K_CORD
#undef SCRATCH_K_CORD
#define SCRATCH_K_CORD(k) ((k) >> 2)
#endif

#ifdef GEMM_D_K
#undef GEMM_D_K
#define GEMM_D_K 768
#endif

GEMM_PARALLEL_FUNC(s8s32dotgemm, int8_t, int32_t, int8_t, int32_t, int32_t,
  8, 12, 12, 12, 0, 0)

