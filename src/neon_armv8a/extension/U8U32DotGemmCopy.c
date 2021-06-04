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
#include "neon_armv8a/I8I32DotGemmCopy.h"

GENERIC_NCOPY_FUNC(u8u32dotgemm, uint8_t, uint32_t, 8)
GENERIC_NCOPY_FUNC(u8u32dotgemm, uint8_t, uint32_t, 12)

TCOPY_FUNC_TEMPLATE(u8u32dotgemm_uint8_t_uint32_t_tcopy_unroll, 8)
TCOPY_FUNC_TEMPLATE(u8u32dotgemm_uint8_t_uint32_t_tcopy_unroll, 12)

