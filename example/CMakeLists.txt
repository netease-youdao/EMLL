#Command line for Android NDK:
# cmake <example_dir> -DANDROID=ON -DANDROID_NDK=/path/to/ndk \
# -DEMLL_DIR=emll/installation/path [-DEML_ARMV7A=ON]
# make

#Command line for GCC:
# cmake <example_dir> [-DCMAKE_SYSROOT=/path/to/gcc/sysroot] \
# -DCMAKE_C_COMPILER=/path/to/gcc \
# -DEMLL_DIR=emll/installation/path [-DEML_ARMV7A=ON]
# make

cmake_minimum_required(VERSION 3.7)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS_RELEASE "-O2")

if(ANDROID) #variable ANDROID_NDK must be provided prior to this section
  set(ANDROID_PLATFORM 27)
  if(EML_ARMV7A)
    set(ANDROID_ABI "armeabi-v7a")
  else() #armv8a
    set(ANDROID_ABI "arm64-v8a")
  endif()
  include(${ANDROID_NDK}/build/cmake/android.toolchain.cmake)
  set(RUNTIME_LIB dl log -fopenmp)
else() #Linux. Variables CMAKE_C_COMPILER must be provided, CMAKE_SYSROOT is optional
  set(CMAKE_SYSTEM_NAME Linux)
  if(EML_ARMV7A)
    set(CMAKE_SYSTEM_PROCESSOR arm)
  else()
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
  endif()
  set(RUNTIME_LIB pthread -fopenmp -lm)
endif()

# variable EMLL_DIR must be provided
project(example_emll C)
include_directories(${EMLL_DIR}/include)
add_executable(example_emll_gemm Gemm.c)
target_link_libraries(example_emll_gemm ${EMLL_DIR}/lib/libeml-armneon.a ${RUNTIME_LIB})

