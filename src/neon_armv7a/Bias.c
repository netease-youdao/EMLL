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


#include "arm_neon/NeonBias.h"
#include "arm_neon/NeonSum.h"

NEON_BIAS(float, float32x4_t, f32, 4, mla)

NEON_BIAS(int32_t, int32x4_t, s32, 4, mla)

NEON_I8I32_SUM(u, uint)

NEON_I16_SUMSQUARE(s, int)

