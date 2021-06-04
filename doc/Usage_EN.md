## Building the Library

### Compilers


| Tested Compilers | ARMv7A | ARMv8A |
| ---------------- | ------ | ------ |
| Linux target| Linaro-GCC-gnueabihf 201912 | Linaro-GCC-aarch64 201912 |
| Android target | NDK-r20 clang | NDK-r20 clang |

### CMake


The CMake version should be 3.7 or newer.


### Linux

A cross-compiling gcc toolchain (7.5.0 or later) is required.


```
git clone https://github.com/netease-youdao/EMLL.git
cd EMLL
mkdir install
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_C_COMPILER=/path/to/gcc [-DCMAKE_SYSROOT=/path/to/toolchain/sysroot] [-DEML_ARMV7A=ON #if built for armv7a 32-bit target]
make install
```

### Android


NDK r19 or newer is required.


```
git clone https://github.com/netease-youdao/EMLL.git
cd EMLL
mkdir install
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install -DANDROID=ON -DANDROID_NDK=/path/to/ndk [-DANDROID_PLATFORM=XX #SDK version of the target device] [-DEML_ARMV7A=ON #if built for armv7a 32-bit target]
make 
make install
```


### Linking with your application


The static library "libeml-armneon.a" will be generated under EMLL/install/lib on building. There are 3 headers under <install_directory>/include (Gemm.h, Quant.h, Layer.h) which summarize the C interfaces provided by the library.

## Testing

When the test option is enabled in cmake, additional executables for testing results and performances will be generated under EMLL/install/bin. They can be executed on the target device with calling from command line (terminal/adb).


| Executable | Command-Line Usage | Notes |
| ---------- | ------------------ | ----- |
| test_gemm | test_gemm < M > < N > < K > <matrix_order> <num_threads> <gemm_type> | matrix_order: 0-3; gemm_type: sgemm, hgemm, u8u32, s8s32 |
| test_bias | test_bias <major_dimension> <minor_dimension> <bias_type> | bias_type: 0-7 for bias, 8-9 for summing of rows/cols |
| test_quant | test_quant <array_size> <job_type> <additional_params> | array_size: the number of elements; job_type: qs/qu/d/rs/ru |

## API

The library provide C functions for GEMM, bias and quantization.


| Functions | Header |
| --------- | ------ |
| General Matrix Multiplication (GEMM) | include/Gemm.h |
| Fully-Connected Layer (FC) with bias | include/Layer.h |
| Quantization, Dequantization, Requantization | include/Quant.h |

### GEMM

For simplicity, the GEMM interface does not include LDA-LDC and alpha (assume 1.0).

The storage order of output matrix C is fixed to column-major. The storage orders of input matrices are specified via function parameters. An element in the matrix can be accessed via column_id "([0, column_numbers))" and row_id "([0, row_numbers))", which can be combined into a 1D index if its storage order is known:

| Storage Order | Element Index |
| ------------- | ------------- |
| Column-Major | column_id * row_numbers + row_id |
| Row-Major | row_id * column_numbers + column_id |

The GEMM interface is summarized in [include/Gemm.h](../include/Gemm.h).

#### Function Name

| Data Types | Function Name |
| ---------- | ------------- |
| fp32 -> fp32 | sgemm |
| fp16 -> fp16 | hgemm <sup>[1] |
| int8 -> int32 | s8s32gemm <sup>[2] |
| uint8 -> uint32 | u8u32gemm <sup>[2] |


[1] Currently not implemented for Aarch32. Return error code 2 when the processor has no support for ARMv8.2a-fp16 ISA


[2] Aarch64 version: Use dot instructions automatically on processors supporting ARMv8.2a-dotprod, use mla-long instructions otherwise


#### Function Parameters

The operation of GEMM: C[MxN] = A[MxK] B[KxN] + beta * C[MxN]

| Parameters | Description |
| ---------- | ----------- |
| a_rowmajor | The storage order of matrix A, row-major if not 0 |
| b_rowmajor | The storage order of matrix B, row-major if not 0 |
| A | The address of the first element in source matrix A |
| B | The address of the first element in source matrix B |
| C | The address of the first element in output matrix C<sup>[1] |
| M | The number of rows in source matrix A |
| N | The number of columns in source matrix B |
| K | The number of columns in A, must be equal to the number of rows in B |
| beta | The scaling factor on C prior to the addition of AB product |
| num_threads | The (maximum) number of threads to use in parallel run |


[1] The output matrix C is fixed to column-major.

### Quantization

Please refer to [include/Quant.h](../include/Quant.h) for details.

#### Function Name


| Name | Description |
| ---- | ----------- |
| bias_int32_t | Perform bias on a 32-bit integer matrix, can be used as a component in asymmetric quantitized GEMM |
| u8u32_sum | Perform row-wise or column-wise sum on the input 8-bit unsigned integer matrix, can be used as a component in asymmetric quantitized GEMM |
| quantize_asymmetric_fX_uY | Asymmetric quantization of X-bit float data to unsigned Y-bit values |
| quantize_symmetric_fX_sY | Symmetric quantization of X-bit float data to signed Y-bit values |
| dequantize_symmetric_fX_sY | Symmetric dequantization of Y-bit integer results to X-bit float ones |
| requantize_asymmetric_XtoY | Asymmetric requantization of X-bit integer values to unsigned Y-bit values |
| requantize_symmetric_XtoY | Symmetric requantization of X-bit integer values to signed Y-bit values |


