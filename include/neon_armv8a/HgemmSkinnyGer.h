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

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n1(const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, float16_t beta_inp);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n2(const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, float16_t beta_inp);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n3(const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, float16_t beta_inp);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n4(const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, float16_t beta_inp);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n5(const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, float16_t beta_inp);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n6(const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, float16_t beta_inp);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n7(const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, float16_t beta_inp);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n8(const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, float16_t beta_inp);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n9(const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, float16_t beta_inp);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n10(const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, float16_t beta_inp);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n11(const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, float16_t beta_inp);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n12(const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, float16_t beta_inp);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n1_omp(
  const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  float16_t beta_inp, uint32_t num_threads);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n2_omp(
  const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  float16_t beta_inp, uint32_t num_threads);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n3_omp(
  const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  float16_t beta_inp, uint32_t num_threads);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n4_omp(
  const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  float16_t beta_inp, uint32_t num_threads);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n5_omp(
  const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  float16_t beta_inp, uint32_t num_threads);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n6_omp(
  const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  float16_t beta_inp, uint32_t num_threads);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n7_omp(
  const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  float16_t beta_inp, uint32_t num_threads);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n8_omp(
  const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  float16_t beta_inp, uint32_t num_threads);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n9_omp(
  const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  float16_t beta_inp, uint32_t num_threads);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n10_omp(
  const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  float16_t beta_inp, uint32_t num_threads);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n11_omp(
  const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  float16_t beta_inp, uint32_t num_threads);

void hgemm_acolmajor_bskinny_afloat16_t_bfloat16_t_n12_omp(
  const float16_t *A, const float16_t *B,
  float16_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  float16_t beta_inp, uint32_t num_threads);

