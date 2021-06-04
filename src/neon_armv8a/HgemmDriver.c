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


#include "neon_armv8a/HgemmKernel.h"
#include "neon_armv8a/HgemmCopy.h"
#include "neon_armv8a/HgemmSkinnyDot.h"
#include "neon_armv8a/HgemmSkinnyGer.h"
#include "arm_neon/ARMCpuType.h"
#include "arm_neon/ARMCompareAndSwap.h"
#include "common/CommonDriver.h"

GEMM_PARALLEL_FUNC(hgemm, float16_t, float16_t, float16_t, float16_t, float16_t,
  8, 16, 12, 12, 12, 12, || blas_arm_get_fp16_support() < 2)

