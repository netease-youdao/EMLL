## 如何借助 CMake 链接和使用 EMLL

### 构建 EMLL

详细步骤请参阅 doc/Usage_ZH.md。

### 在源码中包含 EMLL 的头文件

```
#include "Gemm.h" // 矩阵乘法函数
#include "Layer.h" // 全连接函数
#include "Quant.h" // 量化、反量化、重量化

<其他代码>
```

### 编写 CMakeLists.txt

可以参照 example 文件夹中的 CMakeLists.txt，也可以按如下样式重写：

```
cmake_minimum_required(VERSION <用户指定的最低版本>)
set(CMAKE_BUILD_TYPE <用户指定的构建类型>)

set(CMAKE_C_COMPILER ndk/or/arm-gcc/compiler)
# 添加其他编译选项

project(<用户指定的工程名称> C)

add_executable(<应用程序名> <源文件>)
target_include_directories(<应用程序名> <EMLL安装目录>/include)
target_link_libraries(<应用程序名> <EMLL安装目录>/lib/libeml-armneon.a)

if(ANDROID)
  target_link_libraries(<应用程序名> dl log -fopenmp)
else()
  target_link_libraries(<应用程序名> pthread -fopenmp)
endif()
```

### 构建应用程序

```
cd <CMakeLists.txt 位置>
mkdir build && cd build
cmake .. [-DANDROID=ON #安卓平台] <其他您的工程需要的选项>
make
```

### 示例代码

本文件夹中的 Gemm.c 提供了 EMLL 函数的使用示例，可以通过以下命令编译它并用 adb 拷贝到端侧设备上运行。

```
cd <example 目录>
mkdir build && cd build
cmake .. [-DANDROID=ON -DANDROID_NDK=/path/to/ndk #安卓平台] [-DCMAKE_C_COMPILER=/path/to/gcc [-DCMAKE_SYSROOT=/path/to/gnu/sysroot] #GNU-Linux平台] [-DEML_ARMV7A=ON #armv7平台]
make
# 在 build 文件夹中生成 example_emll_gemm 程序，可到端侧设备上运行它
```

