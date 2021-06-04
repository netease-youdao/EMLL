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


#include "arm_neon/ARMCpuType.h"
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/auxv.h>
#include <asm/hwcap.h>
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include <signal.h>
#include <setjmp.h>

#define MAX_CPU_COUNT 20

struct ARM_CpuType {
  bool m_init;
  uint8_t m_cpuType[MAX_CPU_COUNT];
};

static struct ARM_CpuType blas_arm_cpu_type = {false, {0}};

static pthread_mutex_t blas_arm_get_cpu_type_lock
  = PTHREAD_MUTEX_INITIALIZER;

static bool is_hex(char test) {
  if (test >= 48 && test <= 57) return true; //0-9
  else if (test >= 65 && test <= 70) return true; //A-F
  else if (test >= 97 && test <= 102) return true; //a-f
  else return false;
}

static uint16_t hex2num(char test) {
  if (test >= 48 && test <= 57) return (test - 48); //0-9
  else if (test >= 65 && test <= 70) return (test - 55); //A-F
  else if (test >= 97 && test <= 102) return (test - 87); //a-f
  else return 0;
}

static uint16_t extract_id(const char *line_header,
  unsigned int header_start, unsigned int size) {

  unsigned int header_read = header_start;
  /* find the first colon */
  for (; header_read < size; ++header_read) {
    if (line_header[header_read] == ':') {
      header_read++;
      break;
    }
  }
  /* skip backspace chars after the colon */
  for (; header_read < size; ++header_read) {
    if (line_header[header_read] != ' ') break;
  }
  /* detect 0x or 0X header */
  bool hex_id = false;
  if (header_read + 2 < size) {
    if (line_header[header_read] == '0' &&
      (line_header[header_read + 1] == 'x' || line_header[header_read + 1] == 'X')) {
      hex_id = true;
      header_read += 2;
    }
  }
  /* read number */
  uint16_t id = 0;
  if (hex_id) {
    for (; header_read < size; ++header_read) {
      char test = line_header[header_read];
      if (!is_hex(test)) break;
      id = id * 16 + hex2num(test);
    }
  } else {//decimal
    for (; header_read < size; ++header_read) {
      char test = line_header[header_read];
      if (test < 48 || test > 57) break;
      id = id * 10 + (test - 48);
    }
  }
  return id;
}

/* parse_midr: get CPU model information from MIDR bits */
static uint8_t parse_midr(uint32_t midr) {

  uint8_t cputype = 0; //0 = generic
  uint32_t implementer = midr >> 24;
  uint32_t part = (midr >> 4) & 0xFFF;
  uint32_t variant = (midr >> 20) & 0xF;
  if (implementer == 0x41) { //0x41 == ARM
    if (part == 0xD03) cputype = 53; //Cortex-A53
    else if (part == 0xD04) cputype = 35; //Cortex-A35
    else if (part == 0xD05) {
      if (variant > 0) cputype = 55; //Cortex-A55
      else cputype = 53; //dual-issue ability of Cortex-A55r0 is limited
    }
  }
  else if (implementer == 0x51) { //0x51 == Qualcomm
    if (part == 0x803 || part == 0x801) cputype = 53;
    if (part == 0x805) cputype = 55;
  }
  return cputype;
}

/* MIDR: Main ID Register in ARM processor */
/* direct access of MIDR is not possible in user mode without kernel modules */
/* however the system (Linux/Android) reads MIDR and stores its info to /proc/cpuinfo */
/* so we can assemble the bits of MIDR from the informations in /proc/cpuinfo */
static int read_midr(uint32_t *midr, uint8_t midr_size) {

  FILE *fp = fopen("/proc/cpuinfo", "r");
  if (fp == NULL) {
    return -1; //file open failed
  }

  unsigned char num_cpu_detected = 0;
  unsigned char num_cpu_part_parsed = 0;
  unsigned char num_cpu_vendor_parsed = 0;

  char buffer[300], line_header[30];
  unsigned int header_read = 0, buffer_read = 0;
  bool continue_find_endline = false, line_fill = false;
  size_t bytes_read = 0;
  unsigned int cpuid = 0;
  do {
    bytes_read = fread(buffer, 1, sizeof(buffer), fp);
    if (ferror(fp)) {
      fclose(fp);
      return -2; //error during file read
    }
    for (buffer_read = 0; buffer_read < bytes_read; ) {
      if (continue_find_endline) {
        for (; buffer_read < bytes_read; ++buffer_read) {
          if (buffer[buffer_read] == '\n') {
            continue_find_endline = false;
            buffer_read++;
            break;
          }
        }
      }
      for (; buffer_read < bytes_read; ++buffer_read) {
        if (header_read == sizeof(line_header) || buffer[buffer_read] == '\n') {
          line_fill = true;
          break;
        }
        line_header[header_read] = buffer[buffer_read]; header_read++;
      }
      if (line_fill) {
        for (; header_read < sizeof(line_header); ++header_read) {
          line_header[header_read] = '\0';
        }
        /* extract MIDR information from /proc/cpuinfo */
        /* "CPU implementer : <implementer>" */
        /* "CPU variant     : <variant>" */
        /* "CPU architecture: <architecture>" */
        /* "CPU part        : <part number>" */
        /* "CPU revision    : <revision> */
        if (line_header[0] == 'C' && line_header[1] == 'P' && line_header[2] == 'U'
          && cpuid < midr_size) {

          for (header_read = 3; header_read < sizeof(line_header); ++header_read) {
            if (line_header[header_read] != ' ') break;
          }
          bool skip_detection = false;
          /* extract architecture (MIDR[16:19]) */
          if (header_read + 12 < sizeof(line_header)) {
            if (line_header[header_read] == 'a' && line_header[header_read + 1] == 'r'
              && line_header[header_read + 2] == 'c' && line_header[header_read + 3] == 'h'
              && line_header[header_read + 4] == 'i' && line_header[header_read + 5] == 't') {

              skip_detection = true;
              header_read += 12;
              midr[cpuid] |=
                ((uint32_t)extract_id(line_header, header_read, sizeof(line_header)) << 16);
            }
          }
          /* extract revision (MIDR[0:3]) */
          if (!skip_detection && header_read + 8 < sizeof(line_header)) {
            if (line_header[header_read] == 'r' && line_header[header_read + 1] == 'e'
              && line_header[header_read + 2] == 'v' && line_header[header_read + 3] == 'i'
              && line_header[header_read + 4] == 's' && line_header[header_read + 5] == 'i') {

              skip_detection = true;
              header_read += 8;
              midr[cpuid] |= 
                ((uint32_t)extract_id(line_header, header_read, sizeof(line_header)));
            }
          }
          /* extract variant (MIDR[20:23]) */
          if (!skip_detection && header_read + 7 < sizeof(line_header)) {
            if (line_header[header_read] == 'v' && line_header[header_read + 1] == 'a'
              && line_header[header_read + 2] == 'r' && line_header[header_read + 3] == 'i'
              && line_header[header_read + 4] == 'a' && line_header[header_read + 5] == 'n') {

              skip_detection = true;
              header_read += 7;
              midr[cpuid] |=
                ((uint32_t)extract_id(line_header, header_read, sizeof(line_header)) << 20);
            }
          }
          /* extract implementer (MIDR[24:31]) */
          if (!skip_detection && header_read + 11 < sizeof(line_header)) {
            if (line_header[header_read] == 'i' && line_header[header_read + 1] == 'm'
              && line_header[header_read + 2] == 'p' && line_header[header_read + 3] == 'l'
              && line_header[header_read + 4] == 'e' && line_header[header_read + 5] == 'm') {

              skip_detection = true;
              header_read += 11;
              midr[cpuid] |=
                ((uint32_t)extract_id(line_header, header_read, sizeof(line_header))) << 24;
              num_cpu_vendor_parsed++;
            }
          }
          /* extract part number (MIDR[4:15]) */
          if (!skip_detection && header_read + 4 < sizeof(line_header)) {
            if (line_header[header_read] == 'p' && line_header[header_read + 1] == 'a'
              && line_header[header_read + 2] == 'r' && line_header[header_read + 3] == 't') {

              skip_detection = true;
              header_read += 4;
              midr[cpuid] |=
                ((uint32_t)extract_id(line_header, header_read, sizeof(line_header))) << 4;
              num_cpu_part_parsed++;
            }
          }
        }
        /* read processor id from /proc/cpuinfo */
        /* "processor  : <id>" */
        if (line_header[0] == 'p' && line_header[1] == 'r' && line_header[2] == 'o'
          && line_header[3] == 'c' && line_header[4] == 'e' && line_header[5] == 's'
          && line_header[6] == 's' && line_header[7] == 'o' && line_header[8] == 'r') {

          header_read = 9;
          cpuid = extract_id(line_header, header_read, sizeof(line_header));
          if (cpuid < midr_size) midr[cpuid] = 0;
          num_cpu_detected++;
        }
        line_fill = false;
        header_read = 0;
      }
      for (; buffer_read < bytes_read; ++buffer_read) {
        continue_find_endline = true;
        if (buffer[buffer_read] == '\n') {
          continue_find_endline = false;
          buffer_read++;
          break;
        }
      }
    }
  } while(bytes_read == sizeof(buffer));

  fclose(fp);

  /* on some platforms the Linux kernel is buggy,
   * info from /proc/cpuinfo lack some fields. */
  if (num_cpu_detected != num_cpu_part_parsed) return -3;
  if (num_cpu_detected != num_cpu_vendor_parsed) return -3;
  return num_cpu_detected;
}

static char cpu_uevent[40] = "/sys/devices/system/cpu/cpu";

static uint8_t get_cputype_from_uevent(uint8_t cpuid) {
  /* first form the file path */
  uint8_t digits[8];
  uint8_t n_digits = 0;
  uint8_t tmp = cpuid;
  do {
    digits[n_digits] = tmp % 10;
    tmp /= 10;
    n_digits++;
  } while (tmp > 0);
  for (uint8_t i = 0; i < n_digits; ++i) {
    cpu_uevent[27 + i] = digits[n_digits - i - 1] + 48;
  }
  uint8_t tail_pos = 27 + n_digits;
  cpu_uevent[tail_pos] = '/';
  cpu_uevent[tail_pos + 1] = 'u';
  cpu_uevent[tail_pos + 2] = 'e';
  cpu_uevent[tail_pos + 3] = 'v';
  cpu_uevent[tail_pos + 4] = 'e';
  cpu_uevent[tail_pos + 5] = 'n';
  cpu_uevent[tail_pos + 6] = 't';
  cpu_uevent[tail_pos + 7] = '\0';
  /* then open the file */
  FILE *fp = fopen(cpu_uevent, "r");
  if (fp == NULL) {
    return 0; //file open failed
  }
  unsigned char buffer[100];
  fread(buffer, 1, sizeof(buffer), fp);
  if (ferror(fp)) {
    return 0; //error during read
  }
  uint8_t cputype = 0;
  /* search for patterns like "OF_COMPATIBLE_0=arm,cortex-a72" */
  for (uint8_t i = 0; i < sizeof(buffer) - 40; ++i) {
    if (buffer[i] == 'O' && buffer[i + 1] == 'F' && buffer[i + 2] == '_') {
      i += 3;
      if (buffer[i] == 'C' && buffer[i + 1] == 'O' && buffer[i + 2] == 'M') {
        i += 3;
        if (buffer[i] == 'P' && buffer[i + 1] == 'A' && buffer[i + 2] == 'T') {
          i += 10;
          if (buffer[i] == 'a' && buffer[i + 1] == 'r' && buffer[i + 2] == 'm') {
            i += 4;
            if (buffer[i] == 'c' && buffer[i + 1] == 'o' && buffer[i + 2] == 'r') {
              i += 5;
              if (buffer[i] == 'x' && buffer[i + 1] == '-' && buffer[i + 2] == 'a') {
                char tmp = buffer[i + 3];
                if (tmp >= 48 && tmp <= 57) cputype = tmp - 48;
                tmp = buffer[i + 4];
                if (tmp >= 48 && tmp <= 57) cputype = cputype * 10 + (tmp - 48);
                break;
              }
            }
          }
        }
      }
    }
  }
  return cputype;
}

uint8_t blas_arm_get_cpu_type(uint8_t cpuid) {
  if (cpuid >= MAX_CPU_COUNT) return 0;
  if (!blas_arm_cpu_type.m_init) {
    int acc_lock = pthread_mutex_lock(&blas_arm_get_cpu_type_lock);
    if (acc_lock != 0) return 0;
    if (!blas_arm_cpu_type.m_init) {
      uint32_t midr[MAX_CPU_COUNT];
      for (int cpupos = 0; cpupos < MAX_CPU_COUNT; ++cpupos) {
        midr[cpupos] = 0;
      }
      int midr_read_status = read_midr(midr, MAX_CPU_COUNT);
      if (midr_read_status > MAX_CPU_COUNT) midr_read_status = MAX_CPU_COUNT;
      if (midr_read_status >= 0) {
        for (int cpupos = 0; cpupos < midr_read_status; ++cpupos) {
          blas_arm_cpu_type.m_cpuType[cpupos] =
            parse_midr(midr[cpupos]);
        }
      } else {
        for (int cpupos = 0; cpupos < MAX_CPU_COUNT; ++cpupos) {
          blas_arm_cpu_type.m_cpuType[cpupos] =
            get_cputype_from_uevent(cpupos);
        }
      }
      blas_arm_cpu_type.m_init = true;
    }
    pthread_mutex_unlock(&blas_arm_get_cpu_type_lock);
  }
  return blas_arm_cpu_type.m_cpuType[cpuid];
}

static __thread uint8_t blas_arm_fp16_type = 0;
static __thread uint8_t blas_arm_fp16_init = 0;

#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP (1 << 10)
#endif
#ifndef HWCAP_FPHP
#define HWCAP_FPHP (1 << 9)
#endif

uint8_t blas_arm_get_fp16_support() {
  if (!blas_arm_fp16_init) {
    unsigned long hwcap = getauxval(AT_HWCAP);
#if __aarch64__
    blas_arm_fp16_type =
      ((hwcap & HWCAP_ASIMDHP) && (hwcap & HWCAP_FPHP)) ? 2 : 1;
#else
    blas_arm_fp16_type =
      ((hwcap & HWCAP_VFPv4) && (hwcap & HWCAP_NEON)) ? 1 : 0;
#endif
    blas_arm_fp16_init = 1;
  }
  return blas_arm_fp16_type;
}

#if __aarch64__
#define GEMM_DEFAULT_I8I32_INST 1
#else
#define GEMM_DEFAULT_I8I32_INST 0
#endif

static uint8_t blas_arm_i8i32_type = GEMM_DEFAULT_I8I32_INST + 1;
static uint8_t blas_arm_i8i32_init = 0;
static pthread_mutex_t blas_arm_set_int_lock
  = PTHREAD_MUTEX_INITIALIZER;
static jmp_buf i8i32_ret_env;
static pthread_t int_tid;

static void i8i32gemm_sigill_handler(int sigill) {
  if (pthread_equal(int_tid, pthread_self()) != 0) {
    blas_arm_i8i32_type = GEMM_DEFAULT_I8I32_INST;
    longjmp(i8i32_ret_env, 1);
  } else {
    _Exit(EXIT_FAILURE);
  }
}

static void test_i8i32() {
#if __aarch64__
  __asm__ __volatile__("sdot v1.4s,v0.16b,v2.4b[0]":::"v0","v1","v2");
#else
  __asm__ __volatile__("vmlal.s16 q1,d0,d1[0]":::"q0","q1");
#endif
}

uint8_t blas_arm_get_i8i32_support() {
  if (!blas_arm_i8i32_init) {
    int acc_lock = pthread_mutex_lock(&blas_arm_set_int_lock);
    if (acc_lock != 0) return GEMM_DEFAULT_I8I32_INST;
    if (!blas_arm_i8i32_init) {
      struct sigaction i8i32_act, old_act;
      memset(&i8i32_act, '\0', sizeof(i8i32_act));
      i8i32_act.sa_handler = &i8i32gemm_sigill_handler;
      int_tid = pthread_self();
      if (setjmp(i8i32_ret_env)) {
        sigaction(SIGILL, &old_act, NULL);
        blas_arm_i8i32_init = 1;
        pthread_mutex_unlock(&blas_arm_set_int_lock);
        return GEMM_DEFAULT_I8I32_INST;
      }
      __asm__ __volatile__("dsb sy":::"memory");
      sigaction(SIGILL, &i8i32_act, &old_act);
      test_i8i32();
      sigaction(SIGILL, &old_act, NULL);
      blas_arm_i8i32_init = 1;
    }
    pthread_mutex_unlock(&blas_arm_set_int_lock);
  }
  return blas_arm_i8i32_type;
}

