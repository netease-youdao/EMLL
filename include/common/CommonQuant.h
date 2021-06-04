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


/******************************************************************************
 * File:        CommonQuant.h
 * Description: Function templates for quant/dequant/requant functions.
 *****************************************************************************/

#include <stdint.h>
#include <string.h>

#ifndef INCLUDE_COMMON_QUANT
#define INCLUDE_COMMON_QUANT

/* function template for asymmetric quantization fp -> uint */
#define QUANTIZE_ASYMMETRIC(inbits, outbits) \
void quantize_asymmetric_f##inbits##_u##outbits(\
  const float##inbits##_t *input, uint##outbits##_t *output,\
  uint##outbits##_t *zero_point, float##inbits##_t *scale, uint32_t size,\
  float##inbits##_t input_min, float##inbits##_t input_max) {\
\
  if (size == 0) return;\
  float##inbits##_t min, max;\
  if (input_min <= input_max) {\
    min = input_min;\
    max = input_max;\
  } else {\
    inline_find_extreme_float##inbits##_t(input, size, &min, &max);\
  }\
\
  if (min > 0) min = 0.0;\
  if (max < 0) max = 0.0;\
  const float##inbits##_t max_diff = max - min;\
  if (max_diff == 0.0) {\
    memset(output, 0, size * (outbits >> 3));\
    *zero_point = 0;\
    *scale = 1.0;\
    return;\
  }\
\
  const float##inbits##_t sc = max_diff *\
    (float##inbits##_t)(1.0 / (uint##outbits##_t)-1);\
  *scale = sc;\
  unsigned long long z = ((float##inbits##_t)0.0 - min) / sc\
    + (float##inbits##_t)0.5;\
  const uint##outbits##_t zp = z > (uint##outbits##_t)-1 ?\
    (uint##outbits##_t)-1 : z;\
  *zero_point = zp;\
\
  inline_quant_asym_u##outbits##_from_f##inbits(input, output, size, zp, sc);\
}

/* function template for symmetric quantization fp -> int */
#define QUANTIZE_SYMMETRIC(inbits, outbits) \
void quantize_symmetric_f##inbits##_s##outbits(\
  const float##inbits##_t *input, int##outbits##_t *output,\
  float##inbits##_t *scale, uint32_t size,\
  float##inbits##_t input_min, float##inbits##_t input_max) {\
\
  if (size == 0) return;\
  float##inbits##_t min, max;\
  if (input_min <= input_max) {\
    min = input_min;\
    max = input_max;\
  } else {\
    inline_find_extreme_float##inbits##_t(input, size, &min, &max);\
  }\
\
  const uint##outbits##_t out_abs_max = (uint##outbits##_t)-1 >> 1;\
  const float##inbits##_t sc_positive = max *\
    (float##inbits##_t)(1.0 / out_abs_max);\
  const float##inbits##_t sc_negative = min *\
    (float##inbits##_t)(-1.0 / (out_abs_max + 1));\
  const float##inbits##_t sc =\
    sc_positive > sc_negative ? sc_positive : sc_negative;\
  if (sc == 0.0) {\
    memset(output, 0, size * (outbits >> 3));\
    *scale = 1.0;\
    return;\
  }\
  *scale = sc;\
\
  inline_quant_sym_s##outbits##_from_f##inbits(input, output, size, sc);\
}

/******************************************************************************
 * Template:    REQUANTIZE_ASYMMETRIC_MULHI
 * Description: Function template of asymmetric requantization
 *              based on "mulhi" operations.
 *              Basically, the requantization can be done like this:
 *              (1) determine the min and max of input integers
 *                 if min > 0, min is set to 0
 *                 if max < 0, max is set to 0
 *              (2) calculate scaling factor Sint on input integers:
 *                 Sint = expression_range_of_output_uint / (max - min)
 *              (3) calculate zero point Z of output
 *                 Z = -min * Si
 *              (4) inflate input integers {Ii} to output ints {Oi}
 *                 for i in input index range
 *                   Oi = Ii * Sint + Z
 *              (5) update scaling factor S
 *                 S /= Sint
 *              The steps (1) - (4) are just identical to that in asymmetric
 *              quantization if the inputs are floating numbers. For integers
 *              the situation gets a bit more complicated. The scaling factor
 *              Sint need to be expressed by integer(s). For precision reasons
 *              the exponent and mantissa part of Sint should be stored in
 *              individual integers Bint and Eint:
 *                Sint = (2^exp + mantissa) * (2^-exp) = Bint * (2^-Eint)
 *                Bint = 2^exp + mantissa, Eint = exp
 *              Also, the multiplication Ii * Sint in step (4) changes to
 *              (Ii * Bint) >> Eint.
 *
 *              For integer multiplications on CPU, there're 3 types of
 *              operations normally:
 *              (1) keep all bits of the product, so the length of result
 *                  is twice of that of input
 *              (2) keep only lower half of the product, with output length
 *                  unchanged: "mullo" operation
 *              (3) keep only higher half of the product, with output length
 *                  unchanged: "mulhi" operation
 *              Among the 3 types of operations, type (2) is useful only when
 *              the inputs are small enough (sum of valid bits must be no more
 *              than input length). For type (1), keeping the lower half of
 *              product is not necessary if the input numbers are big enough
 *              (near expression limit). So we choose type (3) for precision
 *              and efficiency concerns.
 *              Generally, we determine a left-shift number L, a mult-factor
 *              M, a right-shift number R and a zero-point Z according to
 *              the min and max of input integers. Then the following steps
 *              are performed on each input integer Ii:
 *              (1) left-shift Ii by L, which can make the min or max number
 *                  approach the expression limit of input integer type,
 *                  so as to minimize the precision loss in subsequent
 *                  "mulhi" operation.
 *              (2) perform "mulhi" operation of shifted Ii with mult-factor
 *                  M to yield (rounded) higher-half product Pi. The value
 *                  of M is also near the expression limit of its type.
 *              (3) right (saturated rounded) shift of Pi by R.
 *                  The right shift is needed to fit results into
 *                  the expression range of output type.
 *              (4) add shifted Pi with Z to get output integer Oi.
 * Parameters:  fp: the type of scale to update (float/double/float16_t/...)
 *              inbits: the number of bits of input integral type
 *              outbits: the number of bits of output integral type
 *              accbits: must be 2 * outbits
 * Dependency:  the following inline functions should be implemented prior
 *              to the introduction of this macro:
 *              (1) inline_find_extreme_int<inbits>_t(
 *                    const int<inbits>_t *dat, uint32_t size,
 *                    int<inbits>_t *min, int<inbits>_t *max) {...}
 *                  This function determines the minimum (write to *min)
 *                  and maximum (write to *max) value of input dat[] which
 *                  has "size" elements.
 *              (2) inline_requant_asym_u<outbits>_from_s<inbits>_mulhi(
 *                    const int<inbits>_t *input, uint<outbits>_t *output,
 *                    uint32_t size, uint8_t L, int<inbits>_t M,
 *                    uint<outbits>_t Z) {...}
 *                  This function performs left-shift on input by L, then
 *                  "mulhi" it with a mult-factor M, right-shift the
 *                  product by R and add it with Z to get output,
 *                  just as the 4 steps shown above.
 *                  The right-shift value R is fixed to accbits-outbits-3
 *                  so it is not in the parameter list.
 *****************************************************************************/
#define REQUANTIZE_ASYMMETRIC_MULHI(fp, inbits, outbits, accbits) \
void requantize_asymmetric_##inbits##to##outbits(\
  const int##inbits##_t *input, uint##outbits##_t *output,\
  fp *scale, uint##outbits##_t *zero_point, uint32_t size,\
  int##inbits##_t input_min, int##inbits##_t input_max) {\
\
  if (size == 0) return;\
  const fp scale_org = *scale;\
  if (scale_org == 0.0) {\
    *zero_point = 0;\
    memset(output, 0, size * sizeof(uint##outbits##_t));\
    return;\
  }\
\
  int##inbits##_t min, max;\
  if (input_min <= input_max) {\
    min = input_min;\
    max = input_max;\
  } else {\
    inline_find_extreme_int##inbits##_t(input, size, &min, &max);\
  }\
  max = max < 0 ? 0 : max;\
  min = min > 0 ? 0 : min;\
  if (min == max) {\
    *zero_point = 0;\
    memset(output, 0, size * sizeof(uint##outbits##_t));\
    return;\
  }\
\
  int##inbits##_t abs_max = -min;\
  if (max > abs_max) abs_max = max;\
  unsigned int max_digits = 0;\
  for (; abs_max > 0; ++max_digits) abs_max >>= 1;\
\
  const int src_lshift = inbits - 1 - max_digits;\
  const uint##inbits##_t range = (uint##inbits##_t)max - (uint##inbits##_t)min;\
\
  uint##accbits##_t mult_par = \
    ((uint##accbits##_t)1 << (accbits - 3)) -\
    ((uint##accbits##_t)1 << (accbits - outbits - 3));\
\
  int##accbits##_t lsh_range = (int##accbits##_t)range << src_lshift;\
  int##inbits##_t mult_factor = mult_par / lsh_range;\
  if (mult_par % lsh_range > lsh_range >> 1) {\
    mult_factor++;\
  }\
\
  int##accbits##_t z_mid = (int##accbits##_t)((-min) << src_lshift) * \
    (int##accbits##_t)mult_factor;\
  int##inbits##_t z_mid2 = z_mid >> (accbits - outbits - 3);\
  if (z_mid & ((int##accbits##_t)1 << (accbits - outbits - 4))) z_mid2++;\
  uint##outbits##_t zp = z_mid2 < 0 ?\
    0 : (z_mid2 > (uint##outbits##_t)-1 ? (uint##outbits##_t)-1 : z_mid2);\
  *zero_point = zp;\
\
  *scale = (*scale) * (fp)range * ((fp)1 / (fp)((uint##outbits##_t)-1));\
  inline_requant_asym_u##outbits##_from_s##inbits##_mulhi(input, output,\
    size, src_lshift, mult_factor, zp);\
}

/******************************************************************************
 * Template:    REQUANTIZE_SYMMETRIC_MULHI
 * Description: Function template of symmetric requantization
 *              based on "mulhi" operations.
 * Parameters:  fp: the type of scale to update (float/double/float16_t/...)
 *              inbits: the number of bits of input integral type
 *              outbits: the number of bits of output integral type
 *              accbits: must be 2 * outbits
 * Dependency:  the following inline functions should be implemented prior
 *              to the introduction of this macro:
 *              (1) inline_find_extreme_int<inbits>_t(
 *                    const int<inbits>_t *dat, uint32_t size,
 *                    int<inbits>_t *min, int<inbits>_t *max) {...}
 *                  This function determines the minimum (write to *min)
 *                  and maximum (write to *max) value of input dat[] which
 *                  has "size" elements.
 *              (2) inline_requant_sym_s<outbits>_from_s<inbits>_mulhi(
 *                    const int<inbits>_t *input, int<outbits>_t *output,
 *                    uint32_t size, uint8_t L, int<inbits>_t M) {...}
 *                  This function performs left-shift on input by L, then
 *                  "mulhi" it with a mult-factor M, finally right-shift
 *                  the product by R to get output,
 *                  The right-shift value R is fixed to accbits-outbits-2
 *                  so it is not in the parameter list.
 *****************************************************************************/
#define REQUANTIZE_SYMMETRIC_MULHI(fp, inbits, outbits, accbits) \
void requantize_symmetric_##inbits##to##outbits(\
  const int##inbits##_t *input, int##outbits##_t *output,\
  fp *scale, uint32_t size,\
  int##inbits##_t input_min, int##inbits##_t input_max) {\
\
  if (size == 0) return;\
  const fp scale_org = *scale;\
  if (scale_org == 0.0) {\
    memset(output, 0, size * sizeof(uint##outbits##_t));\
    return;\
  }\
\
  int##inbits##_t min, max;\
  if (input_min <= input_max) {\
    min = input_min;\
    max = input_max;\
  } else {\
    inline_find_extreme_int##inbits##_t(input, size, &min, &max);\
  }\
  int##inbits##_t max_abs = max;\
  if (max_abs < -min) max_abs = -min;\
  if (max_abs == 0) {\
    memset(output, 0, size * sizeof(uint##outbits##_t));\
    return;\
  }\
\
  int##inbits##_t tmp = max_abs;\
  unsigned int max_digits = 0;\
  for (; tmp > 0; ++max_digits) tmp >>= 1;\
\
  const int src_lshift = inbits - 1 - max_digits;\
  uint##accbits##_t mult_par = \
    ((uint##accbits##_t)1 << (accbits - 3)) -\
    ((uint##accbits##_t)1 << (accbits - outbits - 2));\
  uint##accbits##_t lsh_max_abs = max_abs << src_lshift;\
  int##inbits##_t mult_factor = mult_par / lsh_max_abs;\
  if (mult_par % lsh_max_abs > lsh_max_abs >> 1) {\
    mult_factor++;\
  }\
\
  *scale = (*scale) * (fp)max_abs * ((fp)1 / (fp)(((uint##outbits##_t)-1) >> 1));\
  inline_requant_sym_s##outbits##_from_s##inbits##_mulhi(input, output,\
    size, src_lshift, mult_factor);\
}

#endif
