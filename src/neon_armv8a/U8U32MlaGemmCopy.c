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

#include "common/CommonCopy.h"
#include "arm_neon/NeonI8I32MlaGemmCopy.h"

GENERIC_NCOPY_FUNC(u8u32mlagemm, uint8_t, uint16_t, 8)
GENERIC_NCOPY_FUNC(u8u32mlagemm, uint8_t, uint16_t, 12)

GENERIC_TCOPY_FUNC(u8u32mlagemm, uint8_t, uint16_t, 8)
GENERIC_TCOPY_FUNC(u8u32mlagemm, uint8_t, uint16_t, 12)

