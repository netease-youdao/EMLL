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
 * File:        ARMCompareAndSwap.h
 * Description: Atomic compare and swap functions on ARM processors
 *****************************************************************************/

#include <stdint.h>

/******************************************************************************
 * Function:    atomicCAS_U32
 * Description: Atomic "compare and swap" of 32-bit integer in main memory.
 * Parameters:  comp: the value to compare
 *              write: the value to write
 *              dst: the memory location of 32-bit integer
 * Operation:   # atomic operation
 *              {
 *                uint32_t ret = *dst;
 *                if (*dst == comp) *dst = write;
 *                return ret;
 *              }
 * Return:      The original value of the 32-bit integer in memory
 *****************************************************************************/
uint32_t atomicCAS_U32(uint32_t comp, uint32_t write, uint32_t *dst);

/******************************************************************************
 * Function:    atomicCAS_U64
 * Description: Atomic "compare and swap" of 64-bit integer in main memory.
 * Parameters:  comp: the value to compare
 *              write: the value to write
 *              dst: the memory location of 64-bit integer
 * Operation:   # atomic operation
 *              {
 *                uint64_t ret = *dst;
 *                if (*dst == comp) *dst = write;
 *                return ret;
 *              }
 * Return:      The original value of the 64-bit integer in memory
 *****************************************************************************/
uint64_t atomicCAS_U64(uint64_t comp, uint64_t write, uint64_t *dst);

