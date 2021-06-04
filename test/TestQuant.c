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


#include "common/CommonTest.h"
#include "Quant.h"

TEST_QUANT_UNSYM(32, 8)

TEST_QUANT_SYM(32, 8)

TEST_QUANT_UNSYM(32, 16)

TEST_QUANT_SYM(32, 16)

TEST_DEQUANT_SYM(32, 32)

TEST_REQUANT_UNSYM(float, 32, 8)

TEST_REQUANT_SYM(float, 32, 8)

TEST_REQUANT_UNSYM(float, 32, 16)

TEST_REQUANT_SYM(float, 32, 16)

TEST_REQUANT_UNSYM(float, 16, 8)

TEST_REQUANT_SYM(float, 16, 8)

int main(int argc, char **argv) {

  uint32_t size = argc > 1 ? atoi(argv[1]) : 4;
  const char * const type = argc > 2 ? argv[2] : "qu";

  if (type[0] == 'q') {
    if (type[1] == 'u') {
      if (type[2] == '1') {
        test_quant_asym_f32_u16(size);
      } else {
        test_quant_asym_f32_u8(size);
      }
    } else if (type[1] == 's') {
      if (type[2] == '1') {
        test_quant_sym_f32_s16(size);
      } else {
        test_quant_sym_f32_s8(size);
      }
    }
  } else if (type[0] == 'd') {
    test_dequant_sym_f32_s32(size);
  } else if (type[0] == 'r') {
    if (type[1] == 'u') {
      int32_t max = argc > 3 ? atoi(argv[3]) : 20000000;
      int32_t min = argc > 4 ? atoi(argv[4]) : -10000000;
      float org_scale = argc > 5 ? atof(argv[5]) : 2.0;
      if (type[2] == '1') {
        test_requant_int16_t_float_uint8_t(size,
          (int16_t)(min & 0xFFFF), (int16_t)(max & 0xFFFF), org_scale);
      } else {
        if (type[2] == '3' && type[3] == '2' && type[4] == '1') {
          test_requant_int32_t_float_uint16_t(size, min, max, org_scale);
        } else {
          test_requant_int32_t_float_uint8_t(size, min, max, org_scale);
        }
      }
    } else if (type[1] == 's') {
      uint32_t max_abs = argc > 3 ? atoi(argv[3]) : 2000000;
      float org_scale = argc > 4 ? atof(argv[4]) : 2.0;
      if (type[2] == '1') {
        test_requant_int16_t_float_int8_t(size,
          (uint16_t)(max_abs & 0xFFFF), org_scale);
      } else {
        if (type[2] == '3' && type[3] == '2' && type[4] == '1') {
          test_requant_int32_t_float_int16_t(size, max_abs, org_scale);
        } else {
          test_requant_int32_t_float_int8_t(size, max_abs, org_scale);
        }
      }
    }
  }
  return 0;
}
