## How to link and use EMLL in your application with CMake

### Build EMLL

Please refer to doc/Usage_EN.md for detailed procedure.

### Include Headers in Your Source

```
#include "Gemm.h" // for GEMM functions
#include "Layer.h" // for FC functions
#include "Quant.h" // for quantization/dequantization/requantization

<your code>
```

### Write CMakeLists.txt

You can use the default CMakeLists.txt or manually rewrite it as follows:

```
cmake_minimum_required(VERSION <your minimum version>)
set(CMAKE_BUILD_TYPE <your build type>)

set(CMAKE_C_COMPILER ndk/or/arm-gcc/compiler)
# add your compile options

project(<your project name> C)

add_executable(<your program> <your source file>)
target_include_directories(<your program> <emll_installation_path>/include)
target_link_libraries(<your program> <emll_installation_path>/lib/libeml-armneon.a)

if(ANDROID)
  target_link_libraries(<your program> dl log -fopenmp)
else()
  target_link_libraries(<your program> pthread -fopenmp)
endif()
```

### Build Your Application

```
cd <your_source_dir>
mkdir build && cd build
cmake .. [-DANDROID=ON # for android] [#other options of your project]
make
```

### Example

The source file "Gemm.c" gives an example of using GEMM and quantization functions of EMLL library. It can be built into an executable by the following commands.

```
cd <path/to/emll/example>
mkdir build && cd build
cmake .. [-DANDROID=ON -DANDROID_NDK=/path/to/ndk #options for Android] [-DCMAKE_C_COMPILER=/path/to/gcc [-DCMAKE_SYSROOT=/path/to/gnu/sysroot] #options for GNU-Linux] [-DEML_ARMV7A=ON #armv7 device]
make
# The executable "example_emll_gemm" will be generated under the build directory, which can be executed on the target device.
```

