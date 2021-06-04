# Tuned ARMv8a SGEMM functions for skinny matrices

### Supported shapes and orders
```
C(MxN) = A(MxK) B(KxN)
(1). 4 < M < 51, N >> 50, K >> 50, matrix B is column-major;
(2). 4 < N < 51, M >> 50, K >> 50, matrix A is row-major.
```

### Interface
```
sgemm_skinny1_arowmajor_nXXX_YYY(const float *A, const float *B, float *C,
  uint32_t M, uint32_t K, uint32_t LDA, uint32_t LDB, uint32_t LDC,
  uint8_t b_c_order);

XXX: a number representing the length of dimension N
YYY: letters indicating tuned arm CPU, e.g. a35/a53/a7x
b_c_order: the order of skinny matrices B & C
  0: B & C column-major;
  1: B row-major, C column-major
  2: B column-major, C row-major
  3: B & C row-major
```
