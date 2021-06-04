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


#include "neon_armv8a/sgemm_skinny_dot_kernel/SgemmSkinnyDotKernelA35.h"
#include "neon_armv8a/sgemm_skinny_dot_kernel/SgemmSkinnyDotCopy.h"
#include "neon_armv8a/sgemm_skinny_dot_kernel/SgemmSkinnyDotDriver.h"

DRIVER_PURE_PACK(a35, 4, 10240, 3, 4)
DRIVER_PURE_PACK(a35, 5, 8192, 3, 4)
DRIVER_PURE_PACK(a35, 6, 8192, 3, 4)
DRIVER_PURE_PACK(a35, 7, 6144, 3, 3)
DRIVER_PURE_PACK(a35, 8, 6144, 3, 3)
DRIVER_PURE_PACK(a35, 9, 5120, 4, 4)
DRIVER_PURE_PACK(a35, 10, 5120, 0, 4)
DRIVER_PURE_PACK(a35, 11, 4096, 4, 4)
DRIVER_PURE_PACK(a35, 12, 4096, 0, 4)
DRIVER_PURE_PACK(a35, 13, 3584, 4, 3)
DRIVER_PURE_PACK(a35, 14, 3584, 0, 3)
DRIVER_PURE_PACK(a35, 15, 3072, 4, 3)
DRIVER_PURE_PACK(a35, 16, 3072, 0, 3)
DRIVER_PURE_PACK(a35, 17, 2560, 4, 3)
DRIVER_PURE_PACK(a35, 18, 2560, 4, 3)

DRIVER_MIX2_PACK(a35, 19, 2560, 0, 4, 10, 9, 4)
DRIVER_MIX2_PACK(a35, 20, 2560, 4, 4, 11, 9, 4)
DRIVER_MIX2_PACK(a35, 21, 2048, 0, 4, 12, 9, 4)
DRIVER_MIX2_PACK(a35, 22, 2048, 0, 3, 14, 8, 3)
DRIVER_MIX2_PACK(a35, 23, 2048, 4, 3, 15, 8, 3)
DRIVER_MIX2_PACK(a35, 24, 2048, 0, 3, 16, 8, 3)
DRIVER_MIX2_PACK(a35, 25, 2048, 4, 3, 17, 8, 3)
DRIVER_MIX2_PACK(a35, 26, 1536, 4, 3, 18, 8, 3)
DRIVER_MIX2_PACK(a35, 27, 1536, 0, 4, 14, 13, 3)
DRIVER_MIX2_PACK(a35, 28, 1536, 4, 4, 15, 13, 3)
DRIVER_MIX2_PACK(a35, 29, 1536, 0, 4, 16, 13, 3)
DRIVER_MIX2_PACK(a35, 30, 1536, 4, 4, 17, 13, 3)
DRIVER_MIX2_PACK(a35, 31, 1536, 4, 4, 18, 13, 3)
DRIVER_MIX2_PACK(a35, 32, 1536, 4, 0, 18, 14, 3)
DRIVER_MIX2_PACK(a35, 33, 1536, 4, 4, 18, 15, 3)
DRIVER_MIX2_PACK(a35, 34, 1280, 4, 0, 18, 16, 3)
DRIVER_MIX2_PACK(a35, 35, 1280, 4, 4, 18, 17, 3)
DRIVER_MIX2_PACK(a35, 36, 1280, 4, 4, 18, 18, 3)

DRIVER_MIX3_PACK(a35, 37, 1280, 0, 4, 3, 16, 13, 8, 3)
DRIVER_MIX3_PACK(a35, 38, 1280, 4, 4, 3, 17, 13, 8, 3)
DRIVER_MIX3_PACK(a35, 39, 1280, 4, 4, 3, 18, 13, 8, 3)
DRIVER_MIX3_PACK(a35, 40, 1280, 4, 0, 3, 18, 14, 8, 3)
DRIVER_MIX3_PACK(a35, 41, 1024, 4, 4, 3, 18, 15, 8, 3)
DRIVER_MIX3_PACK(a35, 42, 1024, 4, 0, 3, 18, 16, 8, 3)
DRIVER_MIX3_PACK(a35, 43, 1024, 4, 4, 3, 18, 17, 8, 3)
DRIVER_MIX3_PACK(a35, 44, 1024, 4, 4, 3, 18, 18, 8, 3)
DRIVER_MIX3_PACK(a35, 45, 1024, 4, 0, 4, 18, 14, 13, 3)
DRIVER_MIX3_PACK(a35, 46, 1024, 4, 0, 0, 18, 14, 14, 3)
DRIVER_MIX3_PACK(a35, 47, 1024, 4, 4, 0, 18, 15, 14, 3)
DRIVER_MIX3_PACK(a35, 48, 1024, 4, 4, 4, 18, 15, 15, 3)
DRIVER_MIX3_PACK(a35, 49, 1024, 4, 0, 4, 18, 16, 15, 3)
DRIVER_MIX3_PACK(a35, 50, 1024, 4, 0, 0, 18, 16, 16, 3)


