cmake_minimum_required(VERSION 3.7)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(CMAKE_C_STANDARD 99)
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND NOT CMAKE_C_FLAGS)
  set(CMAKE_C_FLAGS "-O2" CACHE STRING "" FORCE)
endif()

option(EML_ARMV7A "build for armv7a architecture instead of armv8a" OFF)

if(ANDROID)
#Android build
#ANDROID_NDK must be provided
#ANDROID_PLATFORM is optional
  if(NOT DEFINED ANDROID_PLATFORM)
    set(ANDROID_PLATFORM 27)
  endif()
  if(EML_ARMV7A)
    set(ANDROID_ABI "armeabi-v7a")
    set(ANDROID_ARM_MODE arm) #not to use thumb
    set(ANDROID_ARM_NEON ON) #enable NEON on armv7a
  else() #armv8a
    set(ANDROID_ABI "arm64-v8a")
  endif()
  include(${ANDROID_NDK}/build/cmake/android.toolchain.cmake)
  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-g0) #disable NDK debug info generation
  endif()
  set(RUNTIME_LIB dl log)
else()
#Linux build
#CMAKE_C_COMPILER must be provided
#CMAKE_SYSROOT is optional
  set(CMAKE_SYSTEM_NAME Linux)
  if(EML_ARMV7A)
    set(CMAKE_SYSTEM_PROCESSOR arm)
    add_compile_options(-marm -march=armv7ve)
    add_compile_options(-mfpu=neon-vfpv4 -mfp16-format=ieee)
  else()
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
  endif()
  set(RUNTIME_LIB pthread)
endif()

project(emll
 VERSION 1.0
 LANGUAGES C)

include_directories("${PROJECT_SOURCE_DIR}/include")

file(GLOB interface_header "${PROJECT_SOURCE_DIR}/include/*.h")
file(GLOB arm_src "${PROJECT_SOURCE_DIR}/src/arm_neon/*.c")
if(EML_ARMV7A)
  file(GLOB_RECURSE neon_src "${PROJECT_SOURCE_DIR}/src/neon_armv7a/*.c")
  add_library(eml-armneon ${arm_src} ${neon_src})
else()
  file(GLOB neon_src "${PROJECT_SOURCE_DIR}/src/neon_armv8a/*.c")
  file(GLOB skinny_dot_src
    "${PROJECT_SOURCE_DIR}/src/neon_armv8a/sgemm_skinny_dot_kernel/*.c")
  file(GLOB ext_src "${PROJECT_SOURCE_DIR}/src/neon_armv8a/extension/*.c")
  set_source_files_properties(${arm_src} ${ext_src}
    PROPERTIES COMPILE_FLAGS "-march=armv8.2-a+dotprod+fp16")
  add_library(eml-armneon ${arm_src} ${ext_src} ${skinny_dot_src} ${neon_src})
endif()

try_compile(EMLL_COMPILER_OPENMP_SUPPORTED ${CMAKE_BINARY_DIR}
  "${PROJECT_SOURCE_DIR}/test/TestCompilerOpenMP.c"
  COMPILE_DEFINITIONS -fopenmp
  LINK_LIBRARIES -fopenmp)

if (EMLL_COMPILER_OPENMP_SUPPORTED)
  find_package(OpenMP)
  if (OPENMP_C_FOUND)
    target_link_libraries(eml-armneon PUBLIC OpenMP::OpenMP_C)
  else()
    message(STATUS "OpenMP lib not found. Build serial version only.")
    add_definitions(-DEMLL_SERIAL_ONLY)
  endif()
else()
  message(STATUS "The compiler doesn't support OpenMP. Build serial version only.")
  add_definitions(-DEMLL_SERIAL_ONLY)
endif()

option(EML_TEST "build test programs for the library" ON)

if(EML_TEST)
  message(STATUS "Build testing executables for EML")
  set(EML_TEST_EXECUTABLES test_emll_gemm test_emll_bias test_emll_quant)
  add_executable(test_emll_gemm "${PROJECT_SOURCE_DIR}/test/TestGemm.c")
  add_executable(test_emll_bias "${PROJECT_SOURCE_DIR}/test/TestBias.c")
  add_executable(test_emll_quant "${PROJECT_SOURCE_DIR}/test/TestQuant.c")
  target_link_libraries(test_emll_gemm eml-armneon ${RUNTIME_LIB})
  target_link_libraries(test_emll_bias eml-armneon ${RUNTIME_LIB})
  target_link_libraries(test_emll_quant eml-armneon ${RUNTIME_LIB})
endif()

set_target_properties(eml-armneon PROPERTIES PUBLIC_HEADER "${interface_header}")
install(TARGETS eml-armneon ${EML_TEST_EXECUTABLES}
  EXPORT EMLLTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include)

