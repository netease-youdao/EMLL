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

/* In the calculation, the content of skinny matrix B will be read for multiple
 * times. We rearrange its elements to make the reading sequential and
 * contiguous. The process of such rearrangement is called packing. */

/* There are 5 packing types used for skinny matrix B */
/* type_0: row-major contiguous pattern */
/* type_1: partitioned in 4-row chunks, row-major bulk + col-major edge */
/* type_2: partitioned in 2-row chunks, row-major bulk + col-major edge */
/* type_3: partitioned in 2-row chunks, col-major in each chunk */
/* type_4: partitioned in 2-row chunks, x-type interleave like shoelaces */

/* The selection of paking type depends on CPU architecture and problem size */
/* cortex-a35: type_3 when N < 10, type_0 for even N, type_4 for odd N */
/* cortex-a53: type_1 when N < 15, type_2 when 14 < N < 23, type_0 for big N */
/* cortex-a55: the same to cortex-a53 */
/* cortex-a76 & cortex-a72: always type_1 */

/* Example 1 */
/* source matrix B (4x5):
 * a b c d e
 * f g h i j
 * k l m n o
 * p q r s t */
/* pack results to b_scr[] */
/* type_0 pack: abcdefghijklmnopqrst */
/* type_1 pack: abcdfghiklmnpqrsejot */
/* type_2 pack: abcdfghiejklmnpqrsot */
/* type_3 pack: afbgchdiejkplqmrnsot */
/* type_4 pack: agciejfbhdkqmsotplrn */

/* Example 2 */
/* source matrix B (6x6):
 * 11-12-13-14-15-16
 * 21-22-23-24-25-26
 * 31-32-33-34-35-36
 * 41-42-43-44-45-46
 * 51-52-53-54-55-56
 * 61-62-63-64-65-66 */
/* type_0 pack: 11-12-13-14-15-16-21-22-23-24-25-26-31-32-33-34-
 * 35-36-41-42-43-44-45-46-51-52-53-54-55-56-61-62-63-64-65-66 */
/* type_1 pack: 11-12-13-14-21-22-23-24-31-32-33-34-41-42-43-44-
 * 15-25-35-45-16-26-36-46-51-52-53-54-55-56-61-62-63-64-65-66 */
/* type_2 pack: 11-12-13-14-21-22-23-24-15-25-16-26-31-32-33-34-
 * 41-42-43-44-35-45-36-46-51-52-53-54-61-62-63-64-55-65-56-66 */
/* type_3 pack: 11-21-12-22-13-23-14-24-15-25-16-26-31-41-32-42-
 * 33-43-34-44-35-45-36-46-51-61-52-62-53-63-54-64-55-65-56-66 */
/* type-4 pack: 11-22-13-24-15-26-21-12-23-14-25-16-31-42-33-44-
 * 35-46-41-32-43-34-45-36-51-62-53-64-55-66-61-52-63-54-65-56 */

/* type_0 pack from col-major B */
void pack_0_from_cm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N);

/* type_1 pack from col-major B */
void pack_1_from_cm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N);

/* type_2 pack from col-major B */
void pack_2_from_cm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N);

/* type_3 pack from col-major B */
void pack_3_from_cm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N);

/* type_4 pack from col-major B */
void pack_4_from_cm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N);

/* type_0 pack from row-major B */
void pack_0_from_rm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N);

/* type_1 pack from row-major B */
void pack_1_from_rm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N);

/* type_2 pack from row-major B */
void pack_2_from_rm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N);

/* type_3 pack from row-major B */
void pack_3_from_rm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N);

/* type_4 pack from row-major B */
void pack_4_from_rm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N);

