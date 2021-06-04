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
 * File:        ARMCpuType.h
 * Description: Functions support real-time ARM CPU detection:
 *              CPU pipeline type & ISA support.
 *              On ARM, a user program is not allowed to access system
 *              registers holding CPUID information. As a result, the CPU
 *              recognization relies on a "healthy" linux kernel which
 *              read those registers and store their information into sysfs
 *              on initialization process.
 *****************************************************************************/

#include <stdint.h>

/* currently the function can work only on Linux kernels since 2015 */

#ifndef INCLUDE_ARM_CPUTYPE
#define INCLUDE_ARM_CPUTYPE

/*****************************************************************************
 * Function:    blas_arm_get_cpu_type
 * Description: Detect the NEON pipeline type of the CPU. There're 4 major
 *              types of NEON pipelines:
 *              (1) only 1 64-bit NEON pipeline, shared by vector load & arith,
 *                  with in-order execution,
 *                  like that in cortex-A7 and cortex-A35.
 *              (2) 2 64-bit NEON pipelines, can be combined to execute 128-bit
 *                  wide operations, shared by vector load & arith, with
 *                  in-order execution & dual-issue ability,
 *                  like that in cortex-A53.
 *              (3) has identical NEON piplines as stated in (2), with an
 *                  additional load unit capable of simple 64-bit NEON loads
 *                  and element insertion, like that in cortex-A55.
 *              (4) at least 2 64-bit NEON pipelines, out-of-order execution,
 *                  has additional load unit(s) supporting vector loads, like
 *                  that in cortex-A57.
 * Parameter:   cpuid: the ID of CPU core whose type need to be determined,
 *              e.g. the return value of sched_getcpu() when the core where
 *              the calling thread runs needs to be determined.
 * Return:      A 8-bit integer representing the type, 35 for (1), 53 for (2),
 *              55 for (3) and 0 for (4)
 ****************************************************************************/
uint8_t blas_arm_get_cpu_type(uint8_t cpuid);

/*****************************************************************************
 * Function:    blas_arm_get_fp16_support()
 * Description: Determine the support level for half-precision arithmetic
 *              operations of the current system. Rely on "healthy" linux
 *              kernel which detects the CPU correctly.
 * Return:      0 for no-support, 1 for support of conversion from/to fp32,
 *              2 for support of add/mul/fma operations
 ****************************************************************************/
uint8_t blas_arm_get_fp16_support();

/*****************************************************************************
 * Function:    blas_arm_get_i8i32_support()
 * Description: Determine the support level for int8->int32 accumulate
 *              operations of the current system. Rely on "healthy" linux
 *              kernel which detects the CPU correctly.
 * Return:      0 for no-support, 1 for support with *mlal instructions,
 *              2 for support with *dot instructions
 ****************************************************************************/
/* return an integer indicating i8->i32 GEMM support */
/* return 0 for non-support from SIMD */
/* return 1 for basic support with SIMD multiply add */
/* return 2 when armv8.2a-dotprod is available */
uint8_t blas_arm_get_i8i32_support();

#endif
