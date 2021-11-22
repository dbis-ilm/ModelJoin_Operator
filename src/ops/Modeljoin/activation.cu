#include <stdio.h>
#include <assert.h>
#include "cuda_runtime.h"
#include "activation.h"
#include "math.h"
#include "../../utils/cuda_utils.h"

__device__ float relu_activate_kernel(float x){return x*(x>0);}
__device__ float sigmoid_activate_kernel(float x){return 1/(1+exp(-x));}
__device__ float tanh_activate_kernel(float x){return tanh(x);}

__device__ float activate_kernel(float x, TFActivationFunction a)
{
    switch(a){
        case LINEAR:
            return x;
        case RELU:
            return relu_activate_kernel(x);
        case SIGMOID: 
            return sigmoid_activate_kernel(x);
        case TANH:
            return tanh_activate_kernel(x);
    }
    assert(0);
    return 0;
}

__global__ void activate_array_kernel(float *x, int n, TFActivationFunction activation)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) x[i] = activate_kernel(x[i], activation);
}

extern "C" void vector_activate_cuda_unmangled(float *vec, int n, TFActivationFunction activation)
{
    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>(vec, n, activation);
    cuda_check_error(cudaPeekAtLastError());
}

void vector_activate_cuda(float *vec, int n, TFActivationFunction activation)
{
    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>(vec, n, activation);
    cuda_check_error(cudaPeekAtLastError());
}