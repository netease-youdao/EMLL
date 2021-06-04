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

static inline void pref_b(const float *src) {
  __asm__("prfm pldl1keep,[%0,#64]\n\t"::"r"(src):);
}

static inline void pack_rm_from_cm_4col(float * __restrict__ b_wt,
  const float * __restrict__ b_rd, uint32_t K, uint32_t LDB,
  uint32_t N, uint32_t ninc1_2,
  uint32_t ninc1_4, uint32_t ninc2_4, uint32_t ninc3_4) {

  const float *b_l1 = b_rd;
  const float *b_l2 = b_rd + LDB;
  const float *b_l3 = b_rd + LDB * 2;
  const float *b_l4 = b_rd + LDB * 3;
  float *b_w1 = b_wt;

  uint32_t k_left = K;

  for (; k_left > 3; k_left -= 4) {
    float32x4x4_t tmp;
    tmp.val[0] = vld1q_f32(b_l1); b_l1 += 4; pref_b(b_l1);
    tmp.val[1] = vld1q_f32(b_l2); b_l2 += 4; pref_b(b_l2);
    tmp.val[2] = vld1q_f32(b_l3); b_l3 += 4; pref_b(b_l3);
    tmp.val[3] = vld1q_f32(b_l4); b_l4 += 4; pref_b(b_l4);
    vst4q_lane_f32(b_w1, tmp, 0);
    vst4q_lane_f32(b_w1 + ninc1_4, tmp, 1);
    vst4q_lane_f32(b_w1 + ninc2_4, tmp, 2);
    vst4q_lane_f32(b_w1 + ninc3_4, tmp, 3);
    b_w1 += N * 4;
  }
  if (k_left > 1) {
    float32x2x4_t tmp;
    tmp.val[0] = vld1_f32(b_l1); b_l1 += 2;
    tmp.val[1] = vld1_f32(b_l2); b_l2 += 2;
    tmp.val[2] = vld1_f32(b_l3); b_l3 += 2;
    tmp.val[3] = vld1_f32(b_l4); b_l4 += 2;
    vst4_lane_f32(b_w1, tmp, 0);
    vst4_lane_f32(b_w1 + ninc1_2, tmp, 1);
    b_w1 += N * 2;
    k_left -= 2;
  }
  if (k_left > 0) {
    b_w1[0] = *b_l1;
    b_w1[1] = *b_l2;
    b_w1[2] = *b_l3;
    b_w1[3] = *b_l4;
  }
}

void pack_0_from_cm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N) {

  const float *b_rd = B;
  uint32_t n_left = N;
  for (; n_left > 3; n_left -= 4) {
    pack_rm_from_cm_4col(b_scr + N - n_left, b_rd, K, LDB, N,
      N, N, N * 2, N * 3);
    b_rd += 4 * LDB;
  }
  float *b_wt = b_scr + N - n_left;
  if (n_left == 3) {
    const float *b_rd2 = b_rd + LDB;
    const float *b_rd3 = b_rd + LDB * 2;
    uint32_t k_left = K;
    for (; k_left > 3; k_left -= 4) {
      float32x4x3_t tmp;
      tmp.val[0] = vld1q_f32(b_rd); b_rd += 4; pref_b(b_rd);
      tmp.val[1] = vld1q_f32(b_rd2); b_rd2 += 4; pref_b(b_rd2);
      tmp.val[2] = vld1q_f32(b_rd3); b_rd3 += 4; pref_b(b_rd3);
      vst3q_lane_f32(b_wt, tmp, 0); b_wt += N;
      vst3q_lane_f32(b_wt, tmp, 1); b_wt += N;
      vst3q_lane_f32(b_wt, tmp, 2); b_wt += N;
      vst3q_lane_f32(b_wt, tmp, 3); b_wt += N;
    }
    if (k_left > 1) {
      float32x2x3_t tmp;
      tmp.val[0] = vld1_f32(b_rd); b_rd += 2;
      tmp.val[1] = vld1_f32(b_rd2); b_rd2 += 2;
      tmp.val[2] = vld1_f32(b_rd3); b_rd3 += 2;
      vst3_lane_f32(b_wt, tmp, 0); b_wt += N;
      vst3_lane_f32(b_wt, tmp, 1); b_wt += N;
      k_left -= 2;
    }
    if (k_left > 0) {
      b_wt[0] = *b_rd; b_wt[1] = *b_rd2; b_wt[2] = *b_rd3;
    }
  } else if (n_left == 2) {
    const float *b_rd2 = b_rd + LDB;
    uint32_t k_left = K;
    for (; k_left > 3; k_left -= 4) {
      float32x4x2_t tmp;
      tmp.val[0] = vld1q_f32(b_rd); b_rd += 4; pref_b(b_rd);
      tmp.val[1] = vld1q_f32(b_rd2); b_rd2 += 4; pref_b(b_rd2);
      vst2q_lane_f32(b_wt, tmp, 0); b_wt += N;
      vst2q_lane_f32(b_wt, tmp, 1); b_wt += N;
      vst2q_lane_f32(b_wt, tmp, 2); b_wt += N;
      vst2q_lane_f32(b_wt, tmp, 3); b_wt += N;
    }
    if (k_left > 1) {
      float32x2x2_t tmp;
      tmp.val[0] = vld1_f32(b_rd); b_rd += 2;
      tmp.val[1] = vld1_f32(b_rd2); b_rd2 += 2;
      vst2_lane_f32(b_wt, tmp, 0); b_wt += N;
      vst2_lane_f32(b_wt, tmp, 1); b_wt += N;
      k_left -= 2;
    }
    if (k_left > 0) {
      b_wt[0] = *b_rd; b_wt[1] = *b_rd2;
    }
  } else if (n_left == 1) {
    for (uint32_t k_pos = 0; k_pos < K; ++k_pos) {
      *b_wt = b_rd[k_pos]; b_wt += N;
    }
  }
}

void pack_1_from_cm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N) {

  const float *b_rd = B;
  const uint32_t n_4z = N & 0xFFFFFFFC;
  uint32_t n_left = N;
  for (; n_left > 3; n_left -= 4) {
    pack_rm_from_cm_4col(b_scr + N - n_left, b_rd, K, LDB, N,
      N, n_4z, n_4z * 2, n_4z * 3);
    b_rd += 4 * LDB;
  }
  float *b_wt = b_scr + (N - n_left) * 4;
  if (n_left == 3) {
    const float *b_rd2 = b_rd + LDB;
    const float *b_rd3 = b_rd + LDB * 2;
    uint32_t k_left = K;
    for (; k_left > 3; k_left -= 4) {
      float32x4_t tmp1 = vld1q_f32(b_rd); b_rd += 4; pref_b(b_rd);
      float32x4_t tmp2 = vld1q_f32(b_rd2); b_rd2 += 4; pref_b(b_rd2);
      float32x4_t tmp3 = vld1q_f32(b_rd3); b_rd3 += 4; pref_b(b_rd3);
      vst1q_f32(b_wt, tmp1); vst1q_f32(b_wt + 4, tmp2);
      vst1q_f32(b_wt + 8, tmp3); b_wt += N * 4;
    }
    b_wt -= (N - n_left) * 3;
    for (; k_left > 0; k_left--) {
      b_wt[0] = *b_rd++; b_wt[1] = *b_rd2++; b_wt[2] = *b_rd3++;
      b_wt += N;
    }
  } else if (n_left == 2) {
    const float *b_rd2 = b_rd + LDB;
    uint32_t k_left = K;
    for (; k_left > 3; k_left -= 4) {
      float32x4_t tmp1 = vld1q_f32(b_rd); b_rd += 4; pref_b(b_rd);
      float32x4_t tmp2 = vld1q_f32(b_rd2); b_rd2 += 4; pref_b(b_rd2);
      vst1q_f32(b_wt, tmp1); vst1q_f32(b_wt + 4, tmp2);
      b_wt += N * 4;
    }
    b_wt -= (N - n_left) * 3;
    for (; k_left > 0; k_left--) {
      b_wt[0] = *b_rd++; b_wt[1] = *b_rd2++;
      b_wt += N;
    }
  } else if (n_left == 1) {
    uint32_t k_left = K;
    for (; k_left > 3; k_left -= 4) {
      float32x4_t tmp1 = vld1q_f32(b_rd); b_rd += 4; pref_b(b_rd);
      vst1q_f32(b_wt, tmp1); b_wt += N * 4;
    }
    b_wt -= (N - n_left) * 3;
    for (; k_left > 0; k_left--) {
      b_wt[0] = *b_rd++;
      b_wt += N;
    }
  }
}

void pack_2_from_cm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N) {

  const float *b_rd = B;
  const uint32_t n_2z = N & 0xFFFFFFFC;
  uint32_t n_left = N;
  for (; n_left > 3; n_left -= 4) {
    pack_rm_from_cm_4col(b_scr + N - n_left, b_rd, K, LDB, N,
      n_2z, n_2z, N * 2, N * 2 + n_2z);
    b_rd += 4 * LDB;
  }
  float *b_wt = b_scr + (N - n_left) * 2;
  if (n_left == 3) {
    const float *b_rd2 = b_rd + LDB;
    const float *b_rd3 = b_rd + LDB * 2;
    uint32_t k_left = K;
    for (; k_left > 1; k_left -= 2) {
      float32x2_t tmp1 = vld1_f32(b_rd); b_rd += 2;
      float32x2_t tmp2 = vld1_f32(b_rd2); b_rd2 += 2;
      float32x2_t tmp3 = vld1_f32(b_rd3); b_rd3 += 2;
      vst1_f32(b_wt, tmp1); vst1_f32(b_wt + 2, tmp2);
      vst1_f32(b_wt + 4, tmp3); b_wt += N * 2;
    }
    b_wt -= N - n_left;
    if (k_left > 0) {
      b_wt[0] = *b_rd; b_wt[1] = *b_rd2; b_wt[2] = *b_rd3;
    }
  } else if (n_left == 2) {
    const float *b_rd2 = b_rd + LDB;
    uint32_t k_left = K;
    for (; k_left > 1; k_left -= 2) {
      float32x2_t tmp1 = vld1_f32(b_rd); b_rd += 2;
      float32x2_t tmp2 = vld1_f32(b_rd2); b_rd2 += 2;
      vst1_f32(b_wt, tmp1); vst1_f32(b_wt + 2, tmp2);
      b_wt += N * 2;
    }
    b_wt -= N - n_left;
    if (k_left > 0) {
      b_wt[0] = *b_rd; b_wt[1] = *b_rd2;
    }
  } else if (n_left == 1) {
    uint32_t k_left = K;
    for (; k_left > 1; k_left -= 2) {
      float32x2_t tmp1 = vld1_f32(b_rd); b_rd += 2;
      vst1_f32(b_wt, tmp1);
      b_wt += N * 2;
    }
    b_wt -= N - n_left;
    if (k_left > 0) {
      b_wt[0] = *b_rd;
    }
  }
}

void pack_3_from_cm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N) {

  const float *b_rd = B;
  uint32_t n_left = N;
  for (; n_left > 3; n_left -= 4) {
    const float *b_rd1 = b_rd;
    const float *b_rd2 = b_rd + LDB;
    const float *b_rd3 = b_rd + LDB * 2;
    const float *b_rd4 = b_rd2 + LDB * 2;
    float *b_wt = b_scr + (N - n_left) * 2;
    b_rd += LDB * 4;
    uint32_t k_left = K;
    for (; k_left > 3; k_left -= 4) {
      float32x4_t t1 = vld1q_f32(b_rd1); b_rd1 += 4; pref_b(b_rd1);
      float32x4_t t2 = vld1q_f32(b_rd2); b_rd2 += 4; pref_b(b_rd2);
      float32x4_t t3 = vld1q_f32(b_rd3); b_rd3 += 4; pref_b(b_rd3);
      float32x4_t t4 = vld1q_f32(b_rd4); b_rd4 += 4; pref_b(b_rd4);
      vst1_f32(b_wt, vget_low_f32(t1));
      vst1_f32(b_wt + 2, vget_low_f32(t2));
      vst1_f32(b_wt + 4, vget_low_f32(t3));
      vst1_f32(b_wt + 6, vget_low_f32(t4)); b_wt += 2 * N;
      vst1_f32(b_wt, vget_high_f32(t1));
      vst1_f32(b_wt + 2, vget_high_f32(t2));
      vst1_f32(b_wt + 4, vget_high_f32(t3));
      vst1_f32(b_wt + 6, vget_high_f32(t4)); b_wt += 2 * N;
    }
    if (k_left > 1) {
      float32x2_t t1 = vld1_f32(b_rd1); b_rd1 += 2;
      float32x2_t t2 = vld1_f32(b_rd2); b_rd2 += 2;
      float32x2_t t3 = vld1_f32(b_rd3); b_rd3 += 2;
      float32x2_t t4 = vld1_f32(b_rd4); b_rd4 += 2;
      vst1_f32(b_wt, t1); vst1_f32(b_wt + 2, t2);
      vst1_f32(b_wt + 4, t3); vst1_f32(b_wt + 6, t4); b_wt += 2 * N;
      k_left -= 2;
    }
    b_wt -= N - n_left;
    if (k_left > 0) {
      b_wt[0] = *b_rd1; b_wt[1] = *b_rd2;
      b_wt[2] = *b_rd3; b_wt[3] = *b_rd4;
    }
  }
  if (n_left > 1) {
    const float *b_rd1 = b_rd;
    const float *b_rd2 = b_rd + LDB;
    float *b_wt = b_scr + (N - n_left) * 2;
    b_rd += LDB * 2;
    uint32_t k_left = K;
    for (; k_left > 1; k_left -= 2) {
      float32x2_t t1 = vld1_f32(b_rd1); b_rd1 += 2;
      float32x2_t t2 = vld1_f32(b_rd2); b_rd2 += 2;
      vst1_f32(b_wt, t1); vst1_f32(b_wt + 2, t2); b_wt += 2 * N;
    }
    b_wt -= N - n_left;
    if (k_left > 0) {
      b_wt[0] = *b_rd1; b_wt[1] = *b_rd2;
    }
    n_left -= 2;
  }
  if (n_left > 0) {
    float *b_wt = b_scr + (N - n_left) * 2;
    uint32_t k_left = K;
    for (; k_left > 1; k_left -= 2) {
      float32x2_t t1 = vld1_f32(b_rd); b_rd += 2;
      vst1_f32(b_wt, t1); b_wt += 2 * N;
    }
    b_wt -= N - n_left;
    if (k_left > 0) {
      b_wt[0] = *b_rd;
    }
  }
}

void pack_4_from_cm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N) {

  const float *b_rd = B;
  const uint32_t n_2z = (N << 1) - (N & 0xFFFFFFFE);
  uint32_t n_left = N;
  for (; n_left > 3; n_left -= 4) {
    const float *b_rd1 = b_rd;
    const float *b_rd2 = b_rd + LDB;
    const float *b_rd3 = b_rd + LDB * 2;
    const float *b_rd4 = b_rd2 + LDB * 2;
    float *b_wt = b_scr + N - n_left;
    b_rd += LDB * 4;
    uint32_t k_left = K;
    for (; k_left > 3; k_left -= 4) {
      float32x4x4_t tmp;
      tmp.val[0] = vld1q_f32(b_rd1); b_rd1 += 4; pref_b(b_rd1);
      tmp.val[1] = vld1q_f32(b_rd2); b_rd2 += 4; pref_b(b_rd2);
      tmp.val[2] = vld1q_f32(b_rd3); b_rd3 += 4; pref_b(b_rd3);
      tmp.val[3] = vld1q_f32(b_rd4); b_rd4 += 4; pref_b(b_rd4);
      tmp.val[1] = vrev64q_f32(tmp.val[1]);
      tmp.val[3] = vrev64q_f32(tmp.val[3]);
      vst4q_lane_f32(b_wt, tmp, 0);
      vst4q_lane_f32(b_wt + n_2z, tmp, 1); b_wt += 2 * N;
      vst4q_lane_f32(b_wt, tmp, 2);
      vst4q_lane_f32(b_wt + n_2z, tmp, 3); b_wt += 2 * N;
    }
    if (k_left > 1) {
      float32x2_t t1 = vld1_f32(b_rd1); b_rd1 += 2;
      float32x2_t t2 = vld1_f32(b_rd2); b_rd2 += 2;
      float32x2_t t3 = vld1_f32(b_rd3); b_rd3 += 2;
      float32x2_t t4 = vld1_f32(b_rd4); b_rd4 += 2;
      t2 = vrev64_f32(t2); t4 = vrev64_f32(t4);
      float32x2_t d1 = vtrn1_f32(t1, t2);
      float32x2_t d2 = vtrn1_f32(t3, t4);
      float32x2_t d3 = vtrn2_f32(t1, t2);
      float32x2_t d4 = vtrn2_f32(t3, t4);
      vst1_f32(b_wt, d1); vst1_f32(b_wt + 2, d2);
      vst1_f32(b_wt + n_2z, d3); vst1_f32(b_wt + n_2z + 2, d4);
      b_wt += 2 * N; k_left -= 2;
    }
    if (k_left > 0) {
      b_wt[0] = *b_rd1; b_wt[1] = *b_rd2;
      b_wt[2] = *b_rd3; b_wt[3] = *b_rd4;
    }
  }
  if (n_left > 1) {
    const float *b_rd1 = b_rd;
    const float *b_rd2 = b_rd + LDB;
    float *b_wt = b_scr + N - n_left;
    b_rd += LDB * 2;
    uint32_t k_left = K;
    for (; k_left > 1; k_left -= 2) {
      float32x2_t t1 = vld1_f32(b_rd1); b_rd1 += 2;
      float32x2_t t2 = vld1_f32(b_rd2); b_rd2 += 2;
      t2 = vrev64_f32(t2);
      float32x2_t d1 = vtrn1_f32(t1, t2);
      float32x2_t d2 = vtrn2_f32(t1, t2);
      vst1_f32(b_wt, d1); vst1_f32(b_wt + n_2z, d2);
      b_wt += 2 * N;
    }
    if (k_left > 0) {
      b_wt[0] = *b_rd1; b_wt[1] = *b_rd2;
    }
    n_left -= 2;
  }
  if (n_left > 0) {
    float *b_wt = b_scr + N - n_left;
    uint32_t k_left = K;
    for (; k_left > 1; k_left -= 2) {
      float32x2_t t1 = vld1_f32(b_rd); b_rd += 2;
      vst1_f32(b_wt, t1); b_wt += 2 * N;
    }
    if (k_left > 0) {
      b_wt[0] = *b_rd;
    }
  }
}

void pack_0_from_rm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N) {

  uint32_t k_left = K;
  const float *b_rd = B;
  float *b_wt = b_scr;
  for (; k_left > 0; k_left--) {
    const float *b_rd1 = b_rd; b_rd += LDB;
    uint32_t n_left = N;
    for (; n_left > 3; n_left -= 4) {
      float32x4_t t1 = vld1q_f32(b_rd1); b_rd1 += 4;
      vst1q_f32(b_wt, t1); b_wt += 4;
    }
    if (n_left > 1) {
      float32x2_t t1 = vld1_f32(b_rd1); b_rd1 += 2;
      vst1_f32(b_wt, t1); b_wt += 2;
      n_left -= 2;
    }
    if (n_left > 0) {
      *b_wt = *b_rd1; b_wt++;
    }
  }
}

void pack_2_from_rm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N) {

  uint32_t k_left = K;
  const uint32_t n_4z = N & 0xFFFFFFFC;
  const float *b_rd = B;
  float *b_wt = b_scr;
  for (; k_left > 1; k_left -= 2) {
    const float *b_rd1 = b_rd;
    const float *b_rd2 = b_rd + LDB;
    b_rd += LDB * 2;
    float *b_wt1 = b_wt;
    float *b_wt2 = b_wt + n_4z;
    b_wt += N * 2;
    uint32_t n_left = N;
    for (; n_left > 3; n_left -= 4) {
      float32x4_t t1 = vld1q_f32(b_rd1); b_rd1 += 4;
      float32x4_t t2 = vld1q_f32(b_rd2); b_rd2 += 4;
      vst1q_f32(b_wt1, t1); b_wt1 += 4;
      vst1q_f32(b_wt2, t2); b_wt2 += 4;
    }
    for (; n_left > 0; n_left--) {
      b_wt2[0] = *b_rd1++;
      b_wt2[1] = *b_rd2++;
      b_wt2 += 2;
    }
  }
  pack_0_from_rm(b_wt, b_rd, LDB, k_left, N);
}

void pack_1_from_rm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N) {

  uint32_t k_left = K;
  const float *b_rd = B;
  float *b_wt = b_scr;
  const uint32_t n_4z = N & 0xFFFFFFFC;
  for (; k_left > 3; k_left -= 4) {
    const float *b_rd1 = b_rd;
    const float *b_rd2 = b_rd + LDB;
    const float *b_rd3 = b_rd + LDB * 2;
    const float *b_rd4 = b_rd2 + LDB * 2;
    b_rd += LDB * 4;
    float *b_wt1 = b_wt;
    float *b_wt2 = b_wt + n_4z;
    float *b_wt3 = b_wt + n_4z * 2;
    float *b_wt4 = b_wt2 + n_4z * 2;
    b_wt += N * 4;
    uint32_t n_left = N;
    for (; n_left > 3; n_left -= 4) {
      float32x4_t t1 = vld1q_f32(b_rd1); b_rd1 += 4;
      float32x4_t t2 = vld1q_f32(b_rd2); b_rd2 += 4;
      float32x4_t t3 = vld1q_f32(b_rd3); b_rd3 += 4;
      float32x4_t t4 = vld1q_f32(b_rd4); b_rd4 += 4;
      vst1q_f32(b_wt1, t1); b_wt1 += 4;
      vst1q_f32(b_wt2, t2); b_wt2 += 4;
      vst1q_f32(b_wt3, t3); b_wt3 += 4;
      vst1q_f32(b_wt4, t4); b_wt4 += 4;
    }
    for (; n_left > 0; n_left--) {
      b_wt4[0] = *b_rd1++; b_wt4[1] = *b_rd2++;
      b_wt4[2] = *b_rd3++; b_wt4[3] = *b_rd4++; b_wt4 += 4;
    }
  }
  pack_0_from_rm(b_wt, b_rd, LDB, k_left, N);
}

void pack_3_from_rm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N) {

  uint32_t k_left = K;
  const float *b_rd = B;
  float *b_wt = b_scr;
  for (; k_left > 1; k_left -= 2) {
    const float *b_rd1 = b_rd;
    const float *b_rd2 = b_rd + LDB;
    b_rd += LDB * 2;
    float *b_wt1 = b_wt;
    b_wt += N * 2;
    uint32_t n_left = N;
    for (; n_left > 1; n_left -= 2) {
      float32x2_t t1 = vld1_f32(b_rd1); b_rd1 += 2;
      float32x2_t t2 = vld1_f32(b_rd2); b_rd2 += 2;
      float32x2_t d1 = vzip1_f32(t1, t2);
      float32x2_t d2 = vzip2_f32(t1, t2);
      vst1_f32(b_wt1, d1);
      vst1_f32(b_wt1 + 2, d2); b_wt1 += 4;
    }
    if (n_left > 0) {
      b_wt1[0] = *b_rd1; b_wt1[1] = *b_rd2;
    }
  }
  pack_0_from_rm(b_wt, b_rd, LDB, k_left, N);
}

void pack_4_from_rm(float * __restrict__ b_scr,
  const float * __restrict__ B, uint32_t LDB, uint32_t K, uint32_t N) {

  uint32_t k_left = K;
  const float *b_rd = B;
  float *b_wt = b_scr;
  const uint32_t n_2z = (N << 1) - (N & 0xFFFFFFFE);
  for (; k_left > 1; k_left -= 2) {
    const float *b_rd1 = b_rd;
    const float *b_rd2 = b_rd + LDB;
    b_rd += LDB * 2;
    float *b_wt1 = b_wt;
    float *b_wt2 = b_wt + n_2z;
    b_wt += N * 2;
    uint32_t n_left = N;
    for (; n_left > 1; n_left -= 2) {
      float32x2_t t1 = vld1_f32(b_rd1); b_rd1 += 2;
      float32x2_t t2 = vld1_f32(b_rd2); b_rd2 += 2;
      t2 = vrev64_f32(t2);
      float32x2_t d1 = vzip1_f32(t1, t2);
      float32x2_t d2 = vzip2_f32(t2, t1);
      vst1_f32(b_wt1, d1); b_wt1 += 2;
      vst1_f32(b_wt2, d2); b_wt2 += 2;
    }
    if (n_left > 0) {
      b_wt1[0] = *b_rd1; b_wt1[1] = *b_rd2;
    }
  }
  pack_0_from_rm(b_wt, b_rd, LDB, k_left, N);
}

