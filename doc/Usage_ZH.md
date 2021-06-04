## 如何构建 Edge ML 库

### 测试过的编译器


| 端侧设备 | ARMv7A | ARMv8A |
| -------- | ------ | ------ |
| Linux | Linaro-GCC-gnueabihf 201912 | Linaro-GCC-aarch64 201912 |
| Android | NDK-r20 clang | NDK-r20 clang |

目前支持在Linux系统上交叉编译。

### CMake 版本


CMake 需要 3.7 或更新的版本。


### 为运行 Linux 系统的端侧设备构建

需要 7.5.0 及以后的 GCC 交叉编译工具链。

以下为在 Linux 系统开发机上的构建命令

```
git clone https://github.com/netease-youdao/EMLL.git
mkdir install
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_C_COMPILER=GCC编译器的目录 [-DCMAKE_SYSROOT=GCC工具链中sysroot的目录] [-DEML_ARMV7A=ON #若端侧为32位请开此选项]
make
make install
```

### 为运行 Android 系统的端侧设备构建


需要 r19 或更高版本的 Android NDK。

以下为在 Linux 系统开发机上的构建命令

```
git clone https://github.com/netease-youdao/EMLL.git
mkdir install
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DANDROID=ON -DANDROID_NDK=NDK的安装目录 [-DANDROID_PLATFORM=目标安卓SDK版本] [-DEML_ARMV7A=ON #若端侧为32位请开此选项]
make
make install
```


### 使用构建好的库

在 EMLL/install 下会生成 bin，lib 和 include 文件夹，其中 lib 下包含了生成的静态库 libeml-armneon.a，include 下包含了定义 EMLL 对外接口的头文件。应用程序只需在源码中包含对应的头文件，链接时静态链接 libeml-armneon.a 即可。

## 如何测试 Edge ML 库

构建过程中，默认会在 EMLL/install/bin 下生成三个用于测试的可执行文件：test_gemm，test_bias 和 test_quant。把它们拷贝到端侧设备上，命令行 (adb/ssh) 运行它们即可。


| 测试程序  | 命令行参数 | 说明 |
| --------- | ------------------ | ----- |
| test_gemm | test_gemm < M > < N > < K > <源矩阵排列顺序> <并行线程数> <数据类型> | 源矩阵排列顺序：0-3；数据类型：sgemm、hgemm、u8u32、s8s32 |
| test_bias | test_bias <主维度长> <次维度长> <任务种类> | 任务种类：0-7 偏置，8-9 按行或列求和 |
| test_quant | test_quant <测试数组大小> <任务类型> <其他参数> | 任务类型：qs/qu/d/rs/ru |

## 应用程序接口

Edge ML 库提供基于 C 的矩阵乘法和量化接口


| 函数类别 | 头文件 |
| --------- | ------ |
| 矩阵乘法 | include/Gemm.h |
| 全连接层 | include/Layer.h |
| 量化、反量化、重量化 | include/Quant.h |

### 矩阵乘法

为了简便，矩阵乘法接口去掉了 LDA-LDC 参数，固定 alpha = 1.0。

输出矩阵的排列顺序固定为列主序。输入矩阵的排列顺序由函数参数确定。矩阵中的每个元素位置可以通过行号 ([0，行数)) 和列号 ([0，列数)) 确定。当矩阵的排列顺序确定时，其元素地址的偏移量是确定的：

| 排列顺序 | 元素偏移量（相对于首元素）|
| -------- | ------------------------- |
| 列主序 | 列号 * 行数 + 行号 |
| 行主序 | 行号 * 列数 + 列号 |

具体接口定义详见[include/Gemm.h](../include/Gemm.h)。

#### 函数名称

| 数据类型 | 函数名称 |
| ---------- | ------------- |
| fp32 -> fp32 | sgemm |
| fp16 -> fp16 | hgemm <sup>[1] |
| int8 -> int32 | s8s32gemm <sup>[2] |
| uint8 -> uint32 | u8u32gemm <sup>[2] |


[1] 目前不支持 Aarch32 设备；当目标处理器不支持 ARMv8.2-a 半精扩展时，返回错误 2 。


[2] Aarch64 版本：在支持 ARMv8.2a 点积扩展的处理器上自动使用点积指令运算，其他处理器上使用变长乘加指令运算。


#### 函数参数



矩阵乘法通式：C[MxN] = A[MxK] B[KxN] + C[MxN] * beta

| 参数 | 描述 |
| ---------- | ----------- |
| a_rowmajor | 源矩阵 A 的排列顺序，非零表示行主序 |
| b_rowmajor | 源矩阵 B 的排列顺序，非零表示行主序 |
| A | 源矩阵 A 的地址 |
| B | 源矩阵 B 的地址 |
| C | 输出矩阵 C 的地址 |
| M | 矩阵 A 的行数 |
| N | 矩阵 B 的列数 |
| K | A的列数，必须等于 B 的行数 |
| beta | 作用于矩阵 C 的预乘因子 |
| num_threads | 并行时能够使用的线程数 <sup>[2] |


[1] 输出矩阵 C 固定为列主序。


[2] 等于 1 时运行串行版本；等于 0 时使用所有 OpenMP 运行时提供的线程。

### 量化相关函数

详见[include/Quant.h](../include/Quant.h)。

| 函数名 | 描述 |
| ---- | ----------- |
| bias_int32_t | 对32位整数的矩阵施加偏置；可用于非对称量化的整数乘法的后处理 |
| u8u32_sum | 对8位整数的矩阵按行或按列求和，结果存于32位向量 |
| quantize_asymmetric_fX_uY | 非对称量化，从X位浮点到Y位整数 |
| quantize_symmetric_fX_sY | 对称量化，从X位浮点到Y位整数 |
| dequantize_symmetric_fX_sY | 对称反量化，从Y位整数到X位浮点 |
| requantize_asymmetric_XtoY | 非对称重量化，从X位整数到Y位整数 |
| requantize_symmetric_XtoY | 对称重量化，从X位整数到Y位整数 |


