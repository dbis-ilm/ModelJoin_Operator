#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "mj_common.h"
#include "../../operator.h"
#include "../../utils/cuda_utils.h"

extern "C" void dense_layer_consume_tuple_cuda(ModeljoinState *state, int logical_layer, Buffer data, int row) 
{
    long node_in = ((long*)data[Node_in_off])[row];
    long node = ((long*)data[Node_off])[row];
    
    /* We later want to compute x*A, but sgemm only offers A*x. 
    Therefore we save the transposed matrix to compute x*A = A_t * x_t 
    Consequently, the element (node_in, node) is saved at (node, node_in), 
    which translates to node + node_in * dim */
    cuda_check_error(cudaMemcpy(&(state->W_i[logical_layer][node + node_in * state->layer_dims[logical_layer]]), &(((float*)data[W_i_off])[row]),
        sizeof(float), cudaMemcpyHostToDevice));
    cuda_check_error(cudaMemcpy(&(state->b_i[logical_layer][node]), &(((float*)data[b_i_off])[row]),
        sizeof(float), cudaMemcpyHostToDevice));
}

extern "C" void dense_layer_finish_cuda(void *o, int layer) 
{
    Operator *op = (Operator*)o;
    ModeljoinState *state = (ModeljoinState*) op->state;
    int cell_dim = state->layer_dims[layer];
    int i;

    float *new_bias;
    cuda_check_error(cudaMalloc(&new_bias, cell_dim * op->vectorsize * sizeof(float)));

    for (i = 0; i < op->vectorsize; i++) {
        cuda_check_error(cudaMemcpy(&(new_bias[i * cell_dim]), state->b_i[layer], cell_dim * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    cuda_check_error(cudaFree(state->b_i[layer]));

    state->b_i[layer] = new_bias;
}

extern "C" float* dense_layer_forward_matrix_memcpy_loading_cuda(void *o, int layer, float *intermediate, int *int_rows, int *int_cols)
{
    Operator *op = (Operator*)o;
    ModeljoinState *state = (ModeljoinState*) op->state;
    /* cols of the transposed matrix is equal to the vector length */
    int cols = intermediate ? *int_rows : state->num_in_cols;
    /* rows is the layer's dimension and determines the output vector size of the layer */
    int rows = state->layer_dims[layer];
    int vectorsize = op->vectorsize;
    const float alpha = 1;
    const float beta = 1;
    const float gamma = 0;
    float *mat = state->W_i[layer];
    float *bias = state->b_i[layer];
    float *x;
    float *result;
    int i;
    bool allocated = false;
    cublasHandle_t handle = *((cublasHandle_t*)cublas_handle);

    cuda_check_error(cudaMalloc(&result, rows * op->vectorsize * sizeof(float)));

    if (!intermediate) {
        /* Input layer */
        float *transposed;
        cuda_check_error(cudaMalloc(&transposed, cols * vectorsize * sizeof(float)));
        cuda_check_error(cudaMalloc(&x, cols * vectorsize * sizeof(float)));
        allocated = true;
        /* TODO: type */
        for (i = 0; i < cols; i++) {
            cuda_check_error(cudaMemcpy(&(x[i * vectorsize]), ((float*)op->data[state->arg_col_map[i]]), 
                vectorsize * sizeof(float), cudaMemcpyHostToDevice));   
        }
        /* Transpose */
        cublas_check_error(cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, cols, vectorsize, &alpha, x, vectorsize, &gamma, transposed, cols, transposed, cols));
        cuda_check_error(cudaFree(x));
        x = transposed;
    } else {
        x = intermediate;
    }
    cuda_check_error(cudaMemcpy(result, bias, vectorsize * state->layer_dims[layer] * sizeof(float), cudaMemcpyHostToDevice));
    cublas_check_error(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, vectorsize, cols, &alpha, mat, rows, x, cols, &beta, result, rows));
    if (allocated) cuda_check_error(cudaFree(x));

    *int_rows = state->layer_dims[layer];
    *int_cols = vectorsize;
    return result;
}