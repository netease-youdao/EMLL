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

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n1(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, int32_t beta_inp);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n2(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, int32_t beta_inp);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n3(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, int32_t beta_inp);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n4(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, int32_t beta_inp);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n5(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, int32_t beta_inp);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n6(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, int32_t beta_inp);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n7(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, int32_t beta_inp);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n8(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order, int32_t beta_inp);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n1_omp(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  int32_t beta_inp, uint32_t num_threads);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n2_omp(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  int32_t beta_inp, uint32_t num_threads);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n3_omp(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  int32_t beta_inp, uint32_t num_threads);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n4_omp(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  int32_t beta_inp, uint32_t num_threads);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n5_omp(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  int32_t beta_inp, uint32_t num_threads);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n6_omp(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  int32_t beta_inp, uint32_t num_threads);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n7_omp(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  int32_t beta_inp, uint32_t num_threads);

void s8s32mlagemm_arowmajor_bskinny_aint8_t_bint8_t_n8_omp(const int8_t *A, const int8_t *B,
  int32_t *C, uint32_t M, uint32_t K, uint8_t b_c_order,
  int32_t beta_inp, uint32_t num_threads);

