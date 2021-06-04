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

void sgemm_arowmajor_bskinny_afloat_bfloat_n1(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp);

void sgemm_arowmajor_bskinny_afloat_bfloat_n2(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp);

void sgemm_arowmajor_bskinny_afloat_bfloat_n3(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp);

void sgemm_arowmajor_bskinny_afloat_bfloat_n4(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp);

void sgemm_arowmajor_bskinny_afloat_bfloat_n5(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp);

void sgemm_arowmajor_bskinny_afloat_bfloat_n6(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp);

void sgemm_arowmajor_bskinny_afloat_bfloat_n7(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp);

void sgemm_arowmajor_bskinny_afloat_bfloat_n8(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp);

void sgemm_arowmajor_bskinny_afloat_bfloat_n1_omp(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp, uint32_t num_threads);

void sgemm_arowmajor_bskinny_afloat_bfloat_n2_omp(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp, uint32_t num_threads);

void sgemm_arowmajor_bskinny_afloat_bfloat_n3_omp(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp, uint32_t num_threads);

void sgemm_arowmajor_bskinny_afloat_bfloat_n4_omp(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp, uint32_t num_threads);

void sgemm_arowmajor_bskinny_afloat_bfloat_n5_omp(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp, uint32_t num_threads);

void sgemm_arowmajor_bskinny_afloat_bfloat_n6_omp(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp, uint32_t num_threads);

void sgemm_arowmajor_bskinny_afloat_bfloat_n7_omp(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp, uint32_t num_threads);

void sgemm_arowmajor_bskinny_afloat_bfloat_n8_omp(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint8_t b_c_order, float beta_inp, uint32_t num_threads);

