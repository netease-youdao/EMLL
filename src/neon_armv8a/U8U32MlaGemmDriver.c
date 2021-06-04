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


#include "neon_armv8a/U8U32MlaGemmCopy.h"
#include "neon_armv8a/U8U32MlaGemmKernel.h"
#include "neon_armv8a/U8U32MlaGemmSkinnyGer.h"
#include "neon_armv8a/U8U32MlaGemmSkinnyDot.h"
#include "arm_neon/ARMCompareAndSwap.h"
#include "common/CommonDriver.h"

GEMM_PARALLEL_FUNC(u8u32mlagemm, uint8_t, uint16_t, uint8_t, uint16_t, uint32_t,
  8, 12, 8, 8, 8, 8)

