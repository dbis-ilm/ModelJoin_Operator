#include "cublas_v2.h"

#define BLOCK 512

extern void* cublas_handle;

dim3 cuda_gridsize(size_t n);

void cublas_check_error(cublasStatus_t error);

void cuda_check_error(cudaError_t status);

void vector_elementwise_mul_cuda(float *x, float *y, float *res, int n);

void vector_elementwise_add_cuda(float *x, float *y, float *res, int n);