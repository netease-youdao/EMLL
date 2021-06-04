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


#include "common/CommonQuant.h"
#include "arm_neon/NeonQuant.h"

NEON_FIND_EXTREME(float32_t, f32, float32x2_t, float32x4_t, 2)

QUANTIZE_ASYMMETRIC(32, 8)

QUANTIZE_SYMMETRIC(32, 8)

QUANTIZE_ASYMMETRIC(32, 16)

QUANTIZE_SYMMETRIC(32, 16)

void dequantize_symmetric_f32_s32(const int32_t *src, float32_t *dst,
  float32_t scale, uint32_t size) {

  inline_dequant_cvt_f32_s32(dst, src, scale, size);
}

NEON_FIND_EXTREME(int32_t, s32, int32x2_t, int32x4_t, 2)

NEON_FIND_EXTREME(int16_t, s16, int16x4_t, int16x8_t, 4)

REQUANTIZE_ASYMMETRIC_MULHI(float, 32, 8, 64)

REQUANTIZE_SYMMETRIC_MULHI(float, 32, 8, 64)

REQUANTIZE_ASYMMETRIC_MULHI(float, 32, 16, 64)

REQUANTIZE_SYMMETRIC_MULHI(float, 32, 16, 64)

REQUANTIZE_ASYMMETRIC_MULHI(float, 16, 8, 32)

REQUANTIZE_SYMMETRIC_MULHI(float, 16, 8, 32)

