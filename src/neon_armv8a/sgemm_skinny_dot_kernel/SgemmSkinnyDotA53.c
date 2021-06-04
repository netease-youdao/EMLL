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


#include "neon_armv8a/sgemm_skinny_dot_kernel/SgemmSkinnyDotKernelA53.h"
#include "neon_armv8a/sgemm_skinny_dot_kernel/SgemmSkinnyDotCopy.h"
#include "neon_armv8a/sgemm_skinny_dot_kernel/SgemmSkinnyDotDriver.h"

DRIVER_PURE_PACK(a53, 4, 10240, 1, 4)
DRIVER_PURE_PACK(a53, 5, 8192, 1, 4)
DRIVER_PURE_PACK(a53, 6, 8192, 1, 4)
DRIVER_PURE_PACK(a53, 7, 6144, 1, 4)
DRIVER_PURE_PACK(a53, 8, 6144, 1, 4)
DRIVER_PURE_PACK(a53, 9, 5120, 1, 4)
DRIVER_PURE_PACK(a53, 10, 5120, 1, 4)
DRIVER_PURE_PACK(a53, 11, 4096, 1, 4)
DRIVER_PURE_PACK(a53, 12, 4096, 1, 4)
DRIVER_PURE_PACK(a53, 13, 3584, 1, 4)
DRIVER_PURE_PACK(a53, 14, 3584, 1, 4)
DRIVER_PURE_PACK(a53, 15, 3072, 2, 4)
DRIVER_PURE_PACK(a53, 16, 3072, 2, 4)
DRIVER_PURE_PACK(a53, 17, 2048, 2, 4)
DRIVER_PURE_PACK(a53, 18, 2048, 2, 4)
DRIVER_PURE_PACK(a53, 19, 2048, 2, 4)
DRIVER_PURE_PACK(a53, 20, 2048, 2, 4)
DRIVER_PURE_PACK(a53, 21, 2048, 2, 4)
DRIVER_PURE_PACK(a53, 22, 2048, 2, 4)
DRIVER_PURE_PACK(a53, 23, 2048, 0, 4)
DRIVER_PURE_PACK(a53, 24, 2048, 0, 4)
DRIVER_PURE_PACK(a53, 25, 1536, 0, 4)
DRIVER_PURE_PACK(a53, 26, 1536, 0, 4)

DRIVER_MIX2_PACK(a53, 27, 1536, 2, 1, 15, 12, 4)
DRIVER_MIX2_PACK(a53, 28, 1536, 2, 1, 16, 12, 4)
DRIVER_MIX2_PACK(a53, 29, 1536, 2, 1, 17, 12, 4)
DRIVER_MIX2_PACK(a53, 30, 1536, 2, 1, 18, 12, 4)
DRIVER_MIX2_PACK(a53, 31, 1536, 2, 1, 19, 12, 4)
DRIVER_MIX2_PACK(a53, 32, 1536, 2, 1, 20, 12, 4)
DRIVER_MIX2_PACK(a53, 33, 1536, 2, 1, 21, 12, 4)
DRIVER_MIX2_PACK(a53, 34, 1280, 2, 1, 22, 12, 4)
DRIVER_MIX2_PACK(a53, 35, 1280, 0, 1, 23, 12, 4)
DRIVER_MIX2_PACK(a53, 36, 1280, 0, 1, 24, 12, 4)
DRIVER_MIX2_PACK(a53, 37, 1280, 0, 1, 25, 12, 4)
DRIVER_MIX2_PACK(a53, 38, 1280, 0, 1, 26, 12, 4)
DRIVER_MIX2_PACK(a53, 39, 1280, 0, 2, 24, 15, 4)
DRIVER_MIX2_PACK(a53, 40, 1280, 0, 2, 24, 16, 4)
DRIVER_MIX2_PACK(a53, 41, 1024, 0, 2, 24, 17, 4)
DRIVER_MIX2_PACK(a53, 42, 1024, 0, 2, 24, 18, 4)
DRIVER_MIX2_PACK(a53, 43, 1024, 0, 2, 24, 19, 4)
DRIVER_MIX2_PACK(a53, 44, 1024, 0, 2, 24, 20, 4)
DRIVER_MIX2_PACK(a53, 45, 1024, 0, 2, 24, 21, 4)
DRIVER_MIX2_PACK(a53, 46, 1024, 0, 2, 24, 22, 4)
DRIVER_MIX2_PACK(a53, 47, 1024, 0, 0, 24, 23, 4)
DRIVER_MIX2_PACK(a53, 48, 1024, 0, 0, 24, 24, 4)
DRIVER_MIX2_PACK(a53, 49, 1024, 0, 0, 25, 24, 4)
DRIVER_MIX2_PACK(a53, 50, 1024, 0, 0, 26, 24, 4)


