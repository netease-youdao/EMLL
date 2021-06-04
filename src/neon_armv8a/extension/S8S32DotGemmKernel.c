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


#ifdef GEMM_UNSIGNED_INT
#undef GEMM_UNSIGNED_INT
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "common/CommonKernel.h"
#include "arm_neon/ARMCpuType.h"
#include <sched.h>
#include "neon_armv8a/I8I32DotGemmKernel.h"

#define CPUID_DETECT_MNK 1000000

void s8s32dotgemm_kernel_lm_m8n12(uint32_t M, uint32_t N, uint32_t K, int32_t beta,
  const int32_t * __restrict__ sa, const int32_t * __restrict__ sb,
  int32_t * __restrict__ C, uint32_t ldc) {

  uint32_t n_left = N;
  const int32_t *b_head = sb;
  int32_t *c_head = C;
  uint32_t acc_mnk = CPUID_DETECT_MNK;
  uint8_t cpuid = 0, cputype = 0;

  for (; n_left > 11; n_left -= 12) {
    if (acc_mnk >= CPUID_DETECT_MNK) {
      cpuid = sched_getcpu();
      cputype = blas_arm_get_cpu_type(cpuid);
      acc_mnk = 0;
    }
    const int32_t *a_head = sa;
    int32_t *c_ptr = c_head;
    uint32_t m_left = M;
    if (cputype == 55) {
      for (; m_left > 7; m_left -= 8) {
        KERNEL_M8N12_TEMPLATE(A55)
        SAVE_M8N12
        a_head += 8 * K;
        c_ptr += 8;
      }
    } else {
      for (; m_left > 7; m_left -= 8) {
        KERNEL_M8N12_TEMPLATE(A76)
        SAVE_M8N12
        a_head += 8 * K;
        c_ptr += 8;
      }
    }
    MICRO_COMPUTE_LM(4, 12, int32_t, int32_t, int32_t)
    b_head += K * 12;
    c_head += ldc * 12;
    acc_mnk += 12 * K * M;
  }

  ASSEMBLE_DUALPACK_COMPUTE_LM(8, int32_t, int32_t, int32_t, 8)
}

void s8s32dotgemm_kernel_ln_m12n8(uint32_t M, uint32_t N, uint32_t K, int32_t beta,
  const int32_t * __restrict__ sa, const int32_t * __restrict__ sb,
  int32_t * __restrict__ C, uint32_t ldc) {

  uint32_t m_left = M;
  const int32_t *a_head = sa;
  int32_t *c_head = C;
  uint32_t acc_mnk = CPUID_DETECT_MNK;
  uint8_t cpuid = 0, cputype = 0;
  for (; m_left > 11; m_left -= 12) {
    if (acc_mnk >= CPUID_DETECT_MNK) {
      cpuid = sched_getcpu();
      cputype = blas_arm_get_cpu_type(cpuid);
      acc_mnk = 0;
    }
    const int32_t *b_head = sb;
    int32_t *c_ptr = c_head;
    uint32_t n_left = N;
    if (cputype == 55) {
      for (; n_left > 7; n_left -= 8) {
        KERNEL_M12N8_TEMPLATE(A55)
        SAVE_M12N8
        b_head += 8 * K;
        c_ptr += 8 * ldc;
      }
    } else {
      for (; n_left > 7; n_left -= 8) {
        KERNEL_M12N8_TEMPLATE(A76)
        SAVE_M12N8
        b_head += 8 * K;
        c_ptr += 8 * ldc;
      }
    }
    MICRO_COMPUTE_LN(12, 4, int32_t, int32_t, int32_t)
    a_head += K * 12;
    c_head += 12;
    acc_mnk += 12 * N * K;
  }

  ASSEMBLE_DUALPACK_COMPUTE_LN(8, int32_t, int32_t, int32_t, 8)
}

