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

#ifndef INCLUDE_ARM_GEMM_INTERFACE
#define INCLUDE_ARM_GEMM_INTERFACE

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************
Function:    sgemm
Description: fp32 general matrix multiplication, do C = AB + beta * C
             with OpenMP parallelization.
Input: int a_rowmajor: an integer indicating the storage order
                       of input matrix A. Non-zero number for
                       row-major storage, 0 for column-major storage.
       int b_rowmajor: an integer indicating the storage order
                       of input matrix B. Non-zero number for
                       row-major storage, 0 for column-major storage.
       (matrix C is fixed to column-major)
       const float *A, *B: the addresses of input matrices
       uint32_t M, N, K: the dimensions of matrices
                         A: M x K; B: K x N; C: M x N
       float beta: the scale on matrix C prior to GEMM
       uint32_t num_threads: the maximum number of threads
                        in OpenMP parallelization.
                        0 : the function will determine
                            the number of threads from
                            the problem size, use as many
                            threads as possible up to
                            omp_get_max_threads() when
                            M, N and K are large.
                        positive number: limit the maximum
                            number of threads the function
                            can use in OpenMP parallelization
                        1 : force serial execution
Output: float *C: the address of output matrix
Return: 0 on success, 1 on illegal parameters
********************************************************************/
int sgemm(int a_rowmajor, int b_rowmajor,
  const float *A, const float *B, float *C,
  uint32_t M, uint32_t N, uint32_t K,
  float beta, uint32_t num_threads);

/**************************************************************************
Function:    s8s32gemm
Description: signed 8bit -> 32bit integer matrix multiplication,
             do C = AB + beta * C with OpenMP parallelization,
             use *mlal NEON instructions on CPUs without ARMv8.2a feature,
             use *dot NEON instructions on CPUs support ARMv8.2a-dotprod.
Input: int a_rowmajor, b_rowmajor: the same as in function sgemm
       const int8_t *A, *B: the addresses of int8_t input matrices
       M, N, K, beta, num_threads: the same meaning as in function sgemm
Output: int32_t *C: the address of int32_t output matrix C
Return: 0 on success, 1 on illegal parameters
**************************************************************************/
int s8s32gemm(int a_rowmajor, int b_rowmajor,
  const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t N, uint32_t K,
  int32_t beta, uint32_t num_threads);

/**************************************************************************
Function:    u8u32gemm
Description: unsigned 8bit -> 32bit integer matrix multiplication,
             do C = AB with OpenMP parallelization,
             use *mlal NEON instructions on CPUs without ARMv8.2a feature,
             use *dot NEON instructions on CPUs support ARMv8.2a-dotprod.
Input: int a_rowmajor, b_rowmajor: the same as in function sgemm
       const uint8_t *A, *B: the addresses of uint8_t input matrices
       M, N, K, beta, num_threads: the same meaning as in function sgemm
Output: uint32_t *C: the address of uint32_t output matrix C
Return: 0 on success, 1 on illegal parameters
**************************************************************************/
int u8u32gemm(int a_rowmajor, int b_rowmajor,
  const uint8_t *A, const uint8_t *B,
  uint32_t *C, uint32_t M, uint32_t N, uint32_t K,
  uint32_t beta, uint32_t num_threads);

#if __aarch64__
/**************************************************************************
Function:    hgemm
Description: fp16 (half precision) matrix multiplication,
             do C = AB with OpenMP parallelization.
Input: int a_rowmajor, b_rowmajor: the same as in function sgemm
       const float16_t *A, *B: the addresses of input matrices
       M, N, K, beta, num_threads: the same meaning as in function sgemm
Output: float16_t *C: the address of output matrix C
Return: 0 on success, 1 on illegal parameters,
        2 when the CPU doesn't support ARMv8.2a-fp16
**************************************************************************/
int hgemm(int a_rowmajor, int b_rowmajor,
  const float16_t *A, const float16_t *B, float16_t *C,
  uint32_t M, uint32_t N, uint32_t K,
  float16_t beta, uint32_t num_threads);

#endif //aarch64

#ifdef __cplusplus
}
#endif

#endif //INCLUDE_ARM_GEMM_INTERFACE
