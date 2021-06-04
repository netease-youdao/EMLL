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


#include "common/CommonCopy.h"
#include "arm_neon/NeonSgemmCopy.h"

#define NCOPY_float_float(unroll) NCOPY_UNROLL_##unroll

GENERIC_NCOPY_FUNC(sgemm, float, float, 8)
GENERIC_NCOPY_FUNC(sgemm, float, float, 12)

#define TCOPY_UNIT_float_float(src_ptr, dst_ptr, dst_offset, num_elements) \
  TCOPY_UNIT_##num_elements(src_ptr, dst_ptr, dst_offset)

GENERIC_TCOPY_FUNC(sgemm, float, float, 8)
GENERIC_TCOPY_FUNC(sgemm, float, float, 12)
