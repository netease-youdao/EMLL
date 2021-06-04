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


/*************************************************************************
 * File:        CommonSched.h
 * Description: Functions associated with task distribution and
 *              synchronization in parallelized calculations
*************************************************************************/
#include <stdint.h>

#ifndef INCLUDE_COMMON_SCHEDULE
#define INCLUDE_COMMON_SCHEDULE

/* The atomic compare-and-swap instructions are platform-
 * specific, which need to be given elsewhere.
 * If the compiler is GCC, the simplest way is to activate the following
 * macro before including this file */
#ifdef GCC_BUILTIN_SYNC
static uint32_t atomicCAS_U32(uint32_t comp, uint32_t write, uint32_t *dst) {
  __atomic_compare_exchange_n(dst, &comp, write,
    0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  return comp;
}

static uint64_t atomicCAS_U64(uint64_t comp, uint64_t write, uint64_t *dst) {
  __atomic_compare_exchange_n(dst, &comp, write,
    0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
  return comp;
}
#endif

#ifndef D_SCALE
#define D_SCALE 8 //dynamical scaling factor in 2D task distribution
#endif

/******************************************************************************
 * Function:     get_mn_task
 * Description:  Function for a running OpenMP thread to get GEMM task (m, n)
 *               from a (shared) task container atomically.
 * Design:       The task destribution in parallelized GEMM is done in (M, N)
 *               2-dimension space. In parallel run, Several threads are
 *               deployed to compute a MxN output matrix, just like dividing
 *               a rectangular piece of paper with length = N and width = M.
 *               The MxN piece is cut into a number of tiny rectangular pieces
 *               that are distributed to threads. Instead of finishing all the
 *               cutting work before distribution, this function does cutting
 *               and distribution simultaneously. When a thread becomes idle
 *               in parallel zone, it will call this function to cut a new
 *               piece (task) from the remaining paper and take ownership of
 *               the newly-cut small piece(then work on it), or go to a barrier
 *               and wait till other threads finish their work when the paper
 *               has been used up. The size of the newly-get piece(task) is
 *               proportional to the area of the remaining paper.
 *               A typical cutting process is shown below:
 *
 *                 ____________                      ______
 *                |            |                    |      |
 *                |            |  first cut    _____|m1    |
 *             M  |            | -----------> |  n1        |
 *                |            |   m1xn1      |  remain    |
 *                |____________|              |____________|
 *                      N                           |
 *                                     second cut   | m1xn2
 *                                   (m must be m1) |
 *                                                  V    __
 *                                                    m1|  |
 *                 ____________   third cut    _________|  |
 *                |            | <----------- |  n1+n2     |
 *                |M-m1        |     m1xn3    |M-m1        |
 *                |____________|  n3==N-n1-n2 |____________|
 *                      N       (m must be m1)
 *                      |
 *           fourth cut | m2xn4
 *                      |
 *                      V
 *                      _______
 *                     |       |
 *                 ____|       | ---> ---> ---> repeat till nothing left
 *                |____________|
 *
 * Calls:        atomicCAS_U64 (atomic compare and swap, make the cuts atomic)
 * Input:        uint64_t *task_end: the address of a long interger recording
 *                                   the start point of the next task. The
 *                                   long integer represents the coordinates
 *                                   (low-32bit = m, high-32bit = n) of the
 *                                   vortex right above the inward-pointing one
 *                                   in the remaining paper when it is a
 *                                   concave hexagon, or the upper left corner
 *                                   when the remaining is a rectangle.
 *               uint32_t n_pos_min: input the lower bound of N-direction.
 *               uint64_t m_n_pos_max: input the upper bounds
 *                                     along M and N axis
 *                                     low 32 bits: m_max
 *                                     high 32 bits: n_max.
 *               uint64_t m_n_task_min: the minimum task size of
 *                                      m (low 32bit) and n (high 32bit)
 *               uint32_t m_task_max: the maximum task size of m
 *               uint32_t num_threads: input the number of OpenMP threads.
 * Output:       uint32_t *m_start: to output the starting m of the new task.
 *               uint32_t *n_start: to output the starting n of the new task.
 *               uint32_t *m_end: to output the ending m of the new task.
 *               uint32_t *n_end: to output the ending n of the new task.
 * Return:       0 if there's no task left, 1 if a new task has been acquired.
 *****************************************************************************/
static uint32_t get_mn_task(uint64_t *task_end,
  uint32_t *m_start, uint32_t *n_start, uint32_t *m_end, uint32_t *n_end,
  uint64_t m_n_task_min, uint32_t m_task_max,
  uint32_t n_pos_min, uint64_t m_n_pos_max, uint32_t num_threads) {

  const uint32_t m_pos_max = m_n_pos_max & 0xFFFFFFFF;
  const uint32_t n_pos_max = m_n_pos_max >> 32;
  const uint32_t m_task_min_raw = m_n_task_min & 0xFFFFFFFF;
  const uint32_t n_task_min_raw = m_n_task_min >> 32;
  const uint32_t m_task_min = (m_task_min_raw) ? m_task_min_raw : 24;
  const uint32_t n_task_min = (n_task_min_raw) ? n_task_min_raw : 8;

  if (n_pos_max <= n_pos_min) return 0;

  uint32_t mstart, nstart, mend, nend;
  uint64_t task_end_read, task_end_load;

  do {
    task_end_load = *task_end;
    mstart = task_end_load & 0xFFFFFFFF;
    nstart = task_end_load >> 32;

    /* if there is no task left, return 0 */
    if (mstart >= m_pos_max || nstart >= n_pos_max) return 0;

    /* determine how many tasks left in <M, N> 2D space */
    const uint64_t mn_left = (uint64_t)(n_pos_max - n_pos_min) *
      (uint64_t)(m_pos_max - mstart);

    /* determine the msize of the next task */
    /* msize should only depend on mstart, not affected by nstart */
    uint32_t msize = mn_left / (uint64_t)(num_threads * D_SCALE * n_task_min);
    msize = msize / m_task_min * m_task_min;
    if (msize > m_task_max) msize = m_task_max;
    if (msize < m_task_min) msize = m_task_min;
    if (msize > m_pos_max - mstart) msize = m_pos_max - mstart;

    /* determine the nsize of the next task */
    uint32_t n_inc = (nstart >= n_pos_min) ? nstart - n_pos_min : 0;
    uint32_t nsize = (mn_left - (uint64_t)msize * (uint64_t)n_inc) /
      (uint64_t)(num_threads * D_SCALE * msize);
    nsize = nsize / n_task_min * n_task_min;
    if (nsize < n_task_min) nsize = n_task_min;
    if (nsize > n_pos_max - nstart) nsize = n_pos_max - nstart;

    nend = nstart + nsize;
    mend = mstart + msize;
    uint32_t nextm = mstart;
    uint32_t nextn = nend;
    if (nend == n_pos_max) {
      nextm = mend; nextn = n_pos_min;
    }
    uint64_t task_end_write = ((uint64_t)nextn << 32) | (uint64_t)nextm;
    task_end_read = atomicCAS_U64(task_end_load, task_end_write, task_end);
  } while (task_end_read != task_end_load);

  /* write back task info */
  *m_start = mstart;
  *n_start = nstart;
  *m_end = mend;
  *n_end = nend;
  /* if a task has been successfully required, return 1 */
  return 1;
}

/******************************************************************************
 * Function:     get_copy_task
 * Description:  Function for a running thread to get GEMM copy task (m or n)
 *               from a (shared) task container atomically
 * Calls:        atomicCAS_U32
 * Input:        uint32_t *dim_left: the address of an interger recording
 *                                   the amount of remaining work to do,
 *                                   which is shared among threads.
 *               uint32_t min_task: the default size of task to get.
 * Output:       uint32_t *dim_start: to output the starting position
 *                                    of the new task.
 *               uint32_t *dim_end: to output the ending position
 *                                  of the new task.
 * Return:       0 if there's no task left, 1 if a new task has been acquired.
 *****************************************************************************/
static uint32_t get_copy_task(uint32_t *dim_left, uint32_t min_task,
  uint32_t *dim_start, uint32_t *dim_end) {

  if (!min_task) min_task = 24;

  uint32_t dim_left_load, dim_left_read, dim_left_write, dim_get;

  do {
    dim_left_load = *dim_left;
    /* if no task left, return 0 */
    if (dim_left_load == 0) return 0;

    /* determine task size */
    dim_get = dim_left_load % min_task;
    if (dim_get == 0) dim_get = min_task;

    dim_left_write = dim_left_load - dim_get;
    dim_left_read = atomicCAS_U32(dim_left_load, dim_left_write, dim_left);

  } while (dim_left_read != dim_left_load);

  *dim_start = dim_left_write;
  *dim_end = dim_left_load;
  return 1;
}

/******************************************************************************
 * Function:     get_irreg_task
 * Description:  Function for a running thread to get 1D computing task
 *               from a (shared) task container atomically
 * Calls:        atomicCAS_U32
 * Input:        uint32_t *dim_end: the address of an interger recording
 *                                  how much work has been done,
 *                                  which is shared among threads.
 *               uint32_t min_task: specify the default size of a task.
 *               uint32_t max_dim: input the amount of work to do.
 * Output:       uint32_t *task_start: to output the starting position
 *                                     of the new task.
 *               uint32_t *task_end: to output the ending position
 *                                   of the new task.
 * Return:       0 if there's no task left, 1 if a new task has been acquired.
 *****************************************************************************/
static uint32_t get_irreg_task(uint32_t *dim_end,
  uint32_t *task_start, uint32_t *task_end,
  uint32_t min_task, uint32_t max_dim) {

  if (!min_task) min_task = 4;
  uint32_t dim_end_load, dim_end_read, dim_end_write;

  do {
    dim_end_load = *dim_end;
    /* if no task left, return 0 */
    if (dim_end_load >= max_dim) return 0;

    dim_end_write = dim_end_load + min_task;
    if (dim_end_write > max_dim) dim_end_write = max_dim;

    dim_end_read = atomicCAS_U32(dim_end_load, dim_end_write, dim_end);
  } while (dim_end_read != dim_end_load);

  *task_start = dim_end_load;
  *task_end = dim_end_write;
  return 1;
}

#endif

