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
#include <sys/auxv.h>
#include <asm/hwcap.h>

#if __aarch64__
/* detect ARMv8 CAS support */
static __thread uint8_t blas_arm64_cas_type = 0;
static __thread uint8_t blas_arm64_cas_init = 0;

#ifndef HWCAP_ATOMICS
#define HWCAP_ATOMICS (1 << 8)
#endif

static uint8_t blas_arm64_get_cas_support() {
  if (!blas_arm64_cas_init) {
    blas_arm64_cas_type = (getauxval(AT_HWCAP) & HWCAP_ATOMICS) ?
      1 : 0;
    blas_arm64_cas_init = 1;
  }
  return blas_arm64_cas_type;
}

#endif

uint32_t atomicCAS_U32(uint32_t comp, uint32_t write, uint32_t *dst) {
#if __aarch64__
  if (blas_arm64_get_cas_support()) {
    uint32_t tmp = comp;
    __asm__ __volatile__(
      "cas %w0,%w1,[%2]\n\t"
     :"+r"(tmp):"r"(write),"r"(dst):"cc","memory");
    return tmp;
  } else {
    register uint32_t tmp __asm("w0");
    register uint32_t comp_asm __asm("w2") = comp;
    register uint32_t write_asm __asm("w3") = write;
    register uint32_t *dst_asm __asm("x4") = dst;
    __asm__ __volatile__(
      "1:\n\t"
      "ldxr %w0,[%x3]; cmp %w0,%w1; bne 2f; stxr w1,%w2,[%x3]\n\t"
      "cmp w1,#0; bne 1b\n\t"
      "2:\n\t"
     :"+r"(tmp):"r"(comp_asm),"r"(write_asm),"r"(dst_asm):"x1","cc","memory");
    return tmp;
  }
#else
  register uint32_t tmp __asm("r0");
  register uint32_t comp_asm __asm("r2") = comp;
  register uint32_t write_asm __asm("r3") = write;
  register uint32_t *dst_asm __asm("r4") = dst;
  __asm__ __volatile__(
    "1:\n\t"
    "ldrex %0,[%3]; cmp %0,%1; bne 2f; strex r1,%2,[%3]\n\t"
    "cmp r1,#0; bne 1b\n\t"
    "2:\n\t"
    :"+r"(tmp):"r"(comp_asm),"r"(write_asm),"r"(dst_asm):"r1","cc","memory");
  return tmp;
#endif
}

uint64_t atomicCAS_U64(uint64_t comp, uint64_t write, uint64_t *dst) {
  uint64_t tmp;
#if __aarch64__
  if (blas_arm64_get_cas_support()) {
    tmp = comp;
    __asm__ __volatile__(
      "cas %x0,%x1,[%2]\n\t"
     :"+r"(tmp):"r"(write),"r"(dst):"cc","memory");
  } else {
    __asm__ __volatile__(
      "mov x2,%x1; mov x4,%x2\n\t"
      "1:\n\t"
      "ldxr %x0,[%x3]; cmp %x0,x2; bne 2f; stxr w6,x4,[%x3]\n\t"
      "cmp w6,#0; bne 1b\n\t"
      "2:\n\t"
     :"+r"(tmp):"r"(comp),"r"(write),"r"(dst):"x2","x4","w6","cc","memory");
  }
#else
  uint64_t *comp_addr = &comp;
  uint64_t *write_loc = &write;
  uint64_t *tmp_addr = &tmp;
  __asm__ __volatile__(
    "ldr r2,[%0]; ldr r3,[%0,#4]; ldr r4,[%1]; ldr r5,[%1,#4]\n\t"
    "1:\n\t"
    "ldrexd r0,r1,[%2]; cmp r0,r2; bne 2f\n\t"
    "cmp r1,r3; bne 2f; strexd r6,r4,r5,[%2]\n\t"
    "cmp r6,#0; bne 1b\n\t"
    "2:\n\t"
    "str r0,[%3]; str r1,[%3,#4]\n\t"
   ::"r"(comp_addr),"r"(write_loc),"r"(dst),"r"(tmp_addr)
   :"r0","r1","r2","r3","r4","r5","r6","cc","memory");
#endif
  return tmp;
}

