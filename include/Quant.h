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


#include <stdint.h>
#include <arm_neon.h>

#ifndef INCLUDE_ARM_QUANT_INTERFACE
#define INCLUDE_ARM_QUANT_INTERFACE

#ifdef __cplusplus
extern "C" {
#endif

/***********************************************************************
Function:    bias_int32_t
Description: Perform bias operation on a 32-bit signed int matrix.
             This function can be used in asymmetric quantitized GEMM.
Parameters:  dst: the address of the matrix to apply bias on
             bias_dim0: the bias value on every element
             bias_dim1: the address of the input bias vector which
                        will be applied to the matrix along its
                        major dimension, i.e. when the element
                        can be indexed by x * dim1 + y, each element
                        is biased by bias_dim1[y]. No bias will be
                        performed with NULL pointer as input.
             bias_dim1_scale: the scale to be applied on elements
                              of bias_dim1[] prior to the bias
                              operation
             bias_dim2: the address of the input bias vector which
                        whill be applied to the matrix along its
                        minor dimension, i.e. when the element
                        can be indexed by x * dim1 + y, each element
                        is biased by bias_dim2[x]. No bias will be
                        performed with NULL pointer as input.
             bias_dim2_scale: the scale to be applied on elements
                              of bias_dim2[] prior to the bias
                              operation
             dim1: the length of the major dimension of input matrix
             dim2: the length of the minor dimension of input matrix
***********************************************************************/
void bias_int32_t(int32_t *dst, int32_t bias_dim0,
  const int32_t *bias_dim1, int32_t bias_dim1_scale,
  const int32_t *bias_dim2, int32_t bias_dim2_scale,
  uint32_t dim1, uint32_t dim2);

/***********************************************************************
Function:    u8u32_sum
Description: Perform summing operation of cols/rows of the unsigned
             8-bit int matrix. The sum of each col/row is an unsigned
             32-bit integer.
Parameters:  src: the address of input matrix.
             dst: the address of output vector.
             dim1: the length of major dimension of input matrix.
             dim2: the length of minor dimension of input matrix.
             (the major dimension is the vertical one for column-
             major matrix, or the horizontal one for row-major
             matrix)
             direction: the direction of summing
                        0: sum along the minor dimension,
                           output_vector_size == dim1;
                        1: sum along the major dimension,
                           output_vector_size == dim2.
***********************************************************************/
void u8u32_sum(const uint8_t *src, uint32_t *dst,
  uint32_t dim1, uint32_t dim2, uint8_t direction);

/***********************************************************************
Function:    quantize_asymmetric_f32_u8
Description: Asymmetric quantization from fp32 to unsigned 8-bit int,
             producing an 8-bit zero-point integer Z0, a fp32 scale S0
             and quantitized unsigned 8-bit data Q1-Qn on the run.
             For each quantitized element Qi, S0 * (Qi - Z0) can
             approximate the original input (fp32) Fi.
Parameters:  const float32_t *input: the address of the input fp32 array
             uint8_t *output: the address of the output integer array
             uint8_t *zero_point: the address to output Z0
             float32_t *scale: the address to output S0
             uint32_t size: the number of elements in the input
             float32_t input_min, input_max:
                the min and max of input float32_t numbers.
                when input_min > input_max, the min and max
                of input are reevaluated.
***********************************************************************/
void quantize_asymmetric_f32_u8(const float32_t *input, uint8_t *output,
  uint8_t *zero_point, float32_t *scale, uint32_t size,
  float32_t input_min, float32_t input_max);

/***********************************************************************
Function:    quantize_symmetric_f32_s8
Description: symmetric quantization from fp32 to signed 8-bit int,
             producing a fp32 scale S0 and quantitized 8-bit data
             Q1-Qn on the run.
             For each quantitized element Qi, S0 * Qi can
             approximate the original input (fp32) Fi.
Parameters:  const float32_t *input: the address of the input fp32 array
             int8_t *output: the address of the output integer array
             float32_t *scale: the address to output S0
             uint32_t size: the number of elements in the input
             float32_t input_min, input_max:
                the min and max of input float32_t numbers.
                when input_min > input_max, the min and max
                of input are reevaluated.
***********************************************************************/
void quantize_symmetric_f32_s8(const float32_t *input, int8_t *output,
  float32_t *scale, uint32_t size, float32_t input_min, float32_t input_max);

/***********************************************************************
Function:    quantize_asymmetric_f32_u16
Description: Asymmetric quantization from fp32 to unsigned 16-bit int,
             producing an 16-bit zero-point integer Z0, a fp32 scale S0
             and quantitized unsigned 16-bit data Q1-Qn on the run.
             This function does the same thing as
             quantize_asymmetric_f32_u8 except the zero point and
             outputs are 16-bit integers.
***********************************************************************/
void quantize_asymmetric_f32_u16(const float32_t *input, uint16_t *output,
  uint16_t *zero_point, float32_t *scale, uint32_t size,
  float32_t input_min, float32_t input_max);

/***********************************************************************
Function:    quantize_symmetric_f32_s16
Description: symmetric quantization from fp32 to signed 16-bit int,
             producing a fp32 scale S0 and quantitized 16-bit data
             Q1-Qn on the run. This function does the same thing
             as quantize_symmetric_f32_s8 except the outputs are
             16-bit integers.
***********************************************************************/
void quantize_symmetric_f32_s16(const float32_t *input, int16_t *output,
  float32_t *scale, uint32_t size, float32_t input_min, float32_t input_max);

/***********************************************************************
Function:    dequantize_symmetric_f32_s32
Description: Convert 32-bit signed int values to fp32 ones with scaling.
Parameters:  const int32_t *src: the address of the input integer array
             float32_t *dst: the address of the output fp32 array
             float32_t scale: the scaling factor on the input
             uint32_t size: the number of elements in the input
***********************************************************************/
void dequantize_symmetric_f32_s32(const int32_t *src, float32_t *dst,
  float32_t scale, uint32_t size);

/************************************************************************
Function:    requantize_asymmetric_32to8
Description: asymmetric requantization from signed 32-bit int to
             unsigned 8-bit int, which produces an 8-bit zero-point
             integer Z0, updates the fp32 scale S0 and outputs
             requantitized unsigned 8-bit data Q1-Qn on the run.
             For each requantitized element Qi, S0 * (Qi - Z0) can
             approximate the original dequantized value (fp32) Fi
             of the corresponding 32-bit input.
Parameters:  const int32_t *input: the address of the input int array
             uint8_t *output: the address of the output integer array
             float *scale: the address to update scaling factor S0
             uint8_t *zero_point: the address to output Z0
             uint32_t size: the number of elements in the input
             int32_t input_min, input_max: the min and max value
               of input int32 numbers. if input_min > input_max,
               the min and max of the input integers are recalculated.
Note: The following function is near-equivalent to this sequence:
       dequant_cvt_float_int32_t(input, temporal_array, *scale, size);
       quant_unsym_float_uint8_t(temporal_array, output,
         zero_point, scale, size);
************************************************************************/
void requantize_asymmetric_32to8(const int32_t *input, uint8_t *output,
  float *scale, uint8_t *zero_point, uint32_t size,
  int32_t input_min, int32_t input_max);

/************************************************************************
Function:    requantize_symmetric_32to8
Description: symmetric requantization from signed 32-bit int to
             signed 8-bit int, which updates the fp32 scale S0
             and outputs requantitized signed 8-bit data Q1-Qn
             on the run.
             For each requantitized element Qi, S0 * Qi can
             approximate the original dequantized value (fp32) Fi
             of the corresponding 32-bit input.
Parameters:  const int32_t *input: the address of the input int array
             int8_t *output: the address of the output integer array
             float *scale: the address to update scaling factor S0
             uint32_t size: the number of elements in the input
             int32_t input_min, input_max: the min and max value
               of input int32 numbers. if input_min > input_max,
               the min and max of the input integers are recalculated.
Note: The following function is near-equivalent to this sequence:
       dequant_cvt_float_int32_t(input, temporal_array, *scale, size);
       quant_sym_float_int8_t(temporal_array, output, scale, size);
************************************************************************/
void requantize_symmetric_32to8(const int32_t *input, int8_t *output,
  float *scale, uint32_t size,
  int32_t input_min, int32_t input_max);

/************************************************************************
 * Function:    requantize_asymmetric_32to16
 * Description: asymmetric requantization from signed 32-bit int to
 *              unsigned 16-bit int, which does the same thing as
 *              requantize_asymmetric_32to8 except that the outputs
 *              and zero point are 16-bit integers
 ***********************************************************************/
void requantize_asymmetric_32to16(const int32_t *input, uint16_t *output,
  float *scale, uint16_t *zero_point, uint32_t size,
  int32_t input_min, int32_t input_max);

/************************************************************************
 * Function:    requantize_symmetric_32to16
 * Description: symmetric requantization from signed 32-bit int to
 *              signed 16-bit int, which does the same thing as
 *              requantize_symmetric_32to8 except that the outputs
 *              are 16-bit integers
 ***********************************************************************/
void requantize_symmetric_32to16(const int32_t *input, int16_t *output,
  float *scale, uint32_t size,
  int32_t input_min, int32_t input_max);

/************************************************************************
 * Function:    requantize_asymmetric_16to8
 * Description: asymmetric requantization from signed 16-bit int to
 *              unsigned 8-bit int, which does the same thing as
 *              requantize_asymmetric_32to8 except that the inputs
 *              are 16-bit integers
 ***********************************************************************/
void requantize_asymmetric_16to8(const int16_t *input, uint8_t *output,
  float *scale, uint8_t *zero_point, uint32_t size,
  int16_t input_min, int16_t input_max);

/************************************************************************
 * Function:    requantize_symmetric_16to8
 * Description: symmetric requantization from signed 16-bit int to
 *              signed 8-bit int, which does the same thing as
 *              requantize_symmetric_32to8 except that the inputs
 *              are 16-bit integers
 ***********************************************************************/
void requantize_symmetric_16to8(const int16_t *input, int8_t *output,
  float *scale, uint32_t size,
  int16_t input_min, int16_t input_max);

#ifdef __cplusplus
}
#endif

#endif
