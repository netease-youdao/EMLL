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

#ifndef INCLUDE_ARM_LAYER_INTERFACE
#define INCLUDE_ARM_LAYER_INTERFACE

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
Function:       fc
Description:    Function to perform transformation in a fully-connected layer,
                paralleled with OpenMP.
                output = src * weight + bias
Input:          float *src: the address of source data matrix.
                float *weight: the address of weight matrix.
                float *bias: the address of bias vector.
Output:         float *output: the address of output matrix.
Parameters:     int M: the number of rows in source data matrix.
                int K: the number of columns in source data matrix.
                int N: the number of columns in output matrix.
                int trans_src: 1 for column-major source data matrix,
                              0 for row-major source data matrix.
                int trans_weight: 1 for column-major weight matrix,
                                 0 for row-major weight matrix.
                int num_threads: number of OpenMP threads to use.
Return:         0 on success, non-zero number on errors.
******************************************************************************/
int fc(const float *src, const float *weight, const float *bias,
  float *output, int M, int K, int N, int trans_src, int trans_weight,
  int num_threads);

#ifdef __cplusplus
}
#endif

#endif
