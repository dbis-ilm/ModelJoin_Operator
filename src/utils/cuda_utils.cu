#include <assert.h>
#include <stdio.h>
#include "../operator.h"
#include "cuda_utils.h"

void *cublas_handle = NULL;

void cublas_check_error(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return;

        case CUBLAS_STATUS_NOT_INITIALIZED:
            printf("CUBLAS_STATUS_NOT_INITIALIZED\n");

        case CUBLAS_STATUS_ALLOC_FAILED:
            printf("CUBLAS_STATUS_ALLOC_FAILED\n");

        case CUBLAS_STATUS_INVALID_VALUE:
            printf("CUBLAS_STATUS_INVALID_VALUE\n");

        case CUBLAS_STATUS_ARCH_MISMATCH:
            printf("CUBLAS_STATUS_ARCH_MISMATCH\n");

        case CUBLAS_STATUS_MAPPING_ERROR:
            printf("CUBLAS_STATUS_MAPPING_ERROR\n");

        case CUBLAS_STATUS_EXECUTION_FAILED:
            printf("CUBLAS_STATUS_EXECUTION_FAILED\n");

        case CUBLAS_STATUS_INTERNAL_ERROR:
            printf("CUBLAS_STATUS_INTERNAL_ERROR\n");
    }
    assert(0);
}

extern "C" void cublas_init() {
    cublasHandle_t handle;
    cublas_check_error(cublasCreate(&handle));
    cublas_handle = malloc(sizeof(cublasHandle_t));
    *((cublasHandle_t*)cublas_handle) = handle;
}

extern "C" void cublas_exit() {
    cublasDestroy(*((cublasHandle_t*)cublas_handle));
    free(cublas_handle);
}

void error(const char* s)
{
    perror(s);
    assert(0);
    exit(-1);
}

void cuda_check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
    cudaError_t status2 = cudaGetLastError();
    if (status != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error: %s", s);
        error(buffer);
    } 
    if (status2 != cudaSuccess)
    {   
        const char *s = cudaGetErrorString(status);
        char buffer[256];
        printf("CUDA Error Prev: %s\n", s);
        assert(0);
        snprintf(buffer, 256, "CUDA Error Prev: %s", s);
        error(buffer);
    } 
}

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d(x, y, 1);
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

__global__ void elementwise_mul_kernel(float *x, float *y, float *res, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) res[i] = x[i] * y[i];
}

void vector_elementwise_mul_cuda(float *x, float *y, float *res, int n)
{
    elementwise_mul_kernel<<<cuda_gridsize(n), BLOCK>>>(x, y, res, n);
    cuda_check_error(cudaPeekAtLastError());
}

__global__ void elementwise_add_kernel(float *x, float *y, float *res, int n)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) res[i] = x[i] + y[i];
}

void vector_elementwise_add_cuda(float *x, float *y, float *res, int n)
{
    elementwise_add_kernel<<<cuda_gridsize(n), BLOCK>>>(x, y, res, n);
    cuda_check_error(cudaPeekAtLastError());
}