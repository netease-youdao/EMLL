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


#include "neon_armv8a/sgemm_skinny_dot_kernel/SgemmSkinnyDotKernelA7x.h"
#include "neon_armv8a/sgemm_skinny_dot_kernel/SgemmSkinnyDotCopy.h"
#include "neon_armv8a/sgemm_skinny_dot_kernel/SgemmSkinnyDotDriver.h"

DRIVER_PURE_PACK(a7x, 4, 10240, 1, 4)
DRIVER_PURE_PACK(a7x, 5, 8192, 1, 4)
DRIVER_PURE_PACK(a7x, 6, 8192, 1, 4)
DRIVER_PURE_PACK(a7x, 7, 6144, 1, 4)
DRIVER_PURE_PACK(a7x, 8, 6144, 1, 4)
DRIVER_PURE_PACK(a7x, 9, 5120, 1, 4)
DRIVER_PURE_PACK(a7x, 10, 5120, 1, 4)
DRIVER_PURE_PACK(a7x, 11, 4096, 1, 4)
DRIVER_PURE_PACK(a7x, 12, 4096, 1, 4)
DRIVER_PURE_PACK(a7x, 13, 3584, 1, 3)
DRIVER_PURE_PACK(a7x, 14, 3584, 1, 3)
DRIVER_PURE_PACK(a7x, 15, 3072, 1, 3)
DRIVER_PURE_PACK(a7x, 16, 3072, 1, 3)
DRIVER_PURE_PACK(a7x, 17, 2560, 1, 3)
DRIVER_PURE_PACK(a7x, 18, 2560, 1, 3)
DRIVER_PURE_PACK(a7x, 19, 2560, 1, 3)
DRIVER_PURE_PACK(a7x, 20, 2560, 1, 3)
DRIVER_PURE_PACK(a7x, 21, 2048, 1, 3)
DRIVER_PURE_PACK(a7x, 22, 2048, 1, 3)
DRIVER_PURE_PACK(a7x, 23, 2048, 1, 3)
DRIVER_PURE_PACK(a7x, 24, 2048, 1, 3)
DRIVER_PURE_PACK(a7x, 25, 2048, 1, 3)
DRIVER_PURE_PACK(a7x, 26, 1536, 1, 3)

DRIVER_MIX2_PACK(a7x, 27, 1536, 1, 1, 14, 13, 3)
DRIVER_MIX2_PACK(a7x, 28, 1536, 1, 1, 15, 13, 3)
DRIVER_MIX2_PACK(a7x, 29, 1536, 1, 1, 16, 13, 3)
DRIVER_MIX2_PACK(a7x, 30, 1536, 1, 1, 17, 13, 3)
DRIVER_MIX2_PACK(a7x, 31, 1536, 1, 1, 18, 13, 3)
DRIVER_MIX2_PACK(a7x, 32, 1536, 1, 1, 19, 13, 3)
DRIVER_MIX2_PACK(a7x, 33, 1536, 1, 1, 20, 13, 3)
DRIVER_MIX2_PACK(a7x, 34, 1280, 1, 1, 21, 13, 3)
DRIVER_MIX2_PACK(a7x, 35, 1280, 1, 1, 22, 13, 3)
DRIVER_MIX2_PACK(a7x, 36, 1280, 1, 1, 23, 13, 3)
DRIVER_MIX2_PACK(a7x, 37, 1280, 1, 1, 24, 13, 3)
DRIVER_MIX2_PACK(a7x, 38, 1280, 1, 1, 25, 13, 3)
DRIVER_MIX2_PACK(a7x, 39, 1280, 1, 1, 26, 13, 3)
DRIVER_MIX2_PACK(a7x, 40, 1280, 1, 1, 26, 14, 3)
DRIVER_MIX2_PACK(a7x, 41, 1024, 1, 1, 26, 15, 3)
DRIVER_MIX2_PACK(a7x, 42, 1024, 1, 1, 26, 16, 3)
DRIVER_MIX2_PACK(a7x, 43, 1024, 1, 1, 26, 17, 3)
DRIVER_MIX2_PACK(a7x, 44, 1024, 1, 1, 26, 18, 3)
DRIVER_MIX2_PACK(a7x, 45, 1024, 1, 1, 26, 19, 3)
DRIVER_MIX2_PACK(a7x, 46, 1024, 1, 1, 26, 20, 3)
DRIVER_MIX2_PACK(a7x, 47, 1024, 1, 1, 26, 21, 3)
DRIVER_MIX2_PACK(a7x, 48, 1024, 1, 1, 26, 22, 3)
DRIVER_MIX2_PACK(a7x, 49, 1024, 1, 1, 26, 23, 3)
DRIVER_MIX2_PACK(a7x, 50, 1024, 1, 1, 26, 24, 3)


