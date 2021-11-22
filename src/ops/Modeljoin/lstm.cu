#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "mj_common.h"
#include "activation.h"
#include "../../operator.h"
#include "../../utils/cuda_utils.h"

void print_matrix_cuda(float *mat, int rows, int cols)
{
    float *m = (float*) malloc(rows * cols * sizeof(float));
    int i, k;
    cuda_check_error(cudaMemcpy(m, mat, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));

    for (i = 0; i < rows; i++) {
        printf("(");
        for (k = 0; k < cols; k++) {
            printf("%f, ", m[i + k * rows]);
        }
        printf(")\n");
    }
    printf("\n");
    free(m);
}

extern "C" void lstm_layer_consume_tuple_cuda(ModeljoinState *state, int logical_layer, Buffer data, int row) 
{
    long node_in = ((long*)data[Node_in_off])[row];
    long node = ((long*)data[Node_off])[row];

    /* TODO: what if weight is really zero*/
    if (((float*)data[W_i_off])[row] != 0) {
        assert(node == node_in);
        long mat_position = node;
        cuda_check_error(cudaMemcpy(&(state->W_i[logical_layer][mat_position]), &(((float*)data[W_i_off])[row]),
            sizeof(float), cudaMemcpyHostToDevice));
        cuda_check_error(cudaMemcpy(&(state->W_f[logical_layer][mat_position]), &(((float*)data[W_f_off])[row]),
            sizeof(float), cudaMemcpyHostToDevice));
        cuda_check_error(cudaMemcpy(&(state->W_c[logical_layer][mat_position]), &(((float*)data[W_c_off])[row]),
            sizeof(float), cudaMemcpyHostToDevice));
        cuda_check_error(cudaMemcpy(&(state->W_o[logical_layer][mat_position]), &(((float*)data[W_o_off])[row]),
            sizeof(float), cudaMemcpyHostToDevice));
        cuda_check_error(cudaMemcpy(&(state->b_i[logical_layer][node]), &(((float*)data[b_i_off])[row]),
            sizeof(float), cudaMemcpyHostToDevice));
        cuda_check_error(cudaMemcpy(&(state->b_f[logical_layer][node]), &(((float*)data[b_f_off])[row]),
            sizeof(float), cudaMemcpyHostToDevice));
        cuda_check_error(cudaMemcpy(&(state->b_c[logical_layer][node]), &(((float*)data[b_c_off])[row]),
            sizeof(float), cudaMemcpyHostToDevice));
        cuda_check_error(cudaMemcpy(&(state->b_o[logical_layer][node]), &(((float*)data[b_o_off])[row]),
            sizeof(float), cudaMemcpyHostToDevice));
    } else {
        /* We later want to compute x*A, but Intel MKL only offers A*x. 
        Therefore we save the transposed matrix to compute x*A = A_t * x_t 
        Consequently, the element (node_in, node) is saved at (node, node_in), 
        which translates to node + node_in * dim */

        long mat_position = node + node_in * state->layer_dims[logical_layer];
        cuda_check_error(cudaMemcpy(&(state->U_i[logical_layer][mat_position]), &(((float*)data[U_i_off])[row]),
            sizeof(float), cudaMemcpyHostToDevice));
        cuda_check_error(cudaMemcpy(&(state->U_f[logical_layer][mat_position]), &(((float*)data[U_f_off])[row]),
            sizeof(float), cudaMemcpyHostToDevice));
        cuda_check_error(cudaMemcpy(&(state->U_c[logical_layer][mat_position]), &(((float*)data[U_c_off])[row]),
            sizeof(float), cudaMemcpyHostToDevice));
        cuda_check_error(cudaMemcpy(&(state->U_o[logical_layer][mat_position]), &(((float*)data[U_o_off])[row]),
            sizeof(float), cudaMemcpyHostToDevice));
    }
}

extern "C" void lstm_layer_finish_cuda(void *o, int layer) 
{
    Operator *op = (Operator*)o;
    ModeljoinState *state = (ModeljoinState*) op->state;
    int cell_dim = state->layer_dims[layer];
    int i;

    float *new_b_i, *new_b_f, *new_b_c, *new_b_o;
    cuda_check_error(cudaMalloc(&new_b_i, cell_dim * op->vectorsize * sizeof(float)));
    cuda_check_error(cudaMalloc(&new_b_f, cell_dim * op->vectorsize * sizeof(float)));
    cuda_check_error(cudaMalloc(&new_b_c, cell_dim * op->vectorsize * sizeof(float)));
    cuda_check_error(cudaMalloc(&new_b_o, cell_dim * op->vectorsize * sizeof(float)));

    for (i = 0; i < op->vectorsize; i++) {
        cuda_check_error(cudaMemcpy(&(new_b_i[i * cell_dim]), state->b_i[layer], cell_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        cuda_check_error(cudaMemcpy(&(new_b_f[i * cell_dim]), state->b_f[layer], cell_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        cuda_check_error(cudaMemcpy(&(new_b_c[i * cell_dim]), state->b_c[layer], cell_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        cuda_check_error(cudaMemcpy(&(new_b_o[i * cell_dim]), state->b_o[layer], cell_dim * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    cuda_check_error(cudaFree(state->b_i[layer]));
    cuda_check_error(cudaFree(state->b_f[layer]));
    cuda_check_error(cudaFree(state->b_c[layer]));
    cuda_check_error(cudaFree(state->b_o[layer]));

    state->b_i[layer] = new_b_i;
    state->b_f[layer] = new_b_f;
    state->b_c[layer] = new_b_c;
    state->b_o[layer] = new_b_o;
}

extern "C" float* lstm_layer_forward_cuda(void *o, int layer, float *intermediate, int *int_rows, int *int_cols)
{
    Operator *op = (Operator*)o;
    ModeljoinState *state = (ModeljoinState*) op->state;
    cublasHandle_t handle = *((cublasHandle_t*)cublas_handle);

    /* TODO: Rough assumption here */
    int num_recurrence = state->num_in_cols;
    int round;

    /* Calculation based on https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py LSTMCell:call() */
    
    int cell_dim = state->layer_dims[layer];
    int vectorsize = op->vectorsize;
    float *h_tm1 = NULL; /* Memory State, vector of size cell_dim*/
    float *c_tm1 = NULL; /* Cell State, vector size cell_dim */
    
    float *z_i, *z_f, *z_c, *z_o, *data;
    cuda_check_error(cudaMalloc(&z_i, vectorsize * cell_dim * sizeof(float)));
    cuda_check_error(cudaMalloc(&z_f, vectorsize * cell_dim * sizeof(float)));
    cuda_check_error(cudaMalloc(&z_c, vectorsize * cell_dim * sizeof(float)));
    cuda_check_error(cudaMalloc(&z_o, vectorsize * cell_dim * sizeof(float)));
    cuda_check_error(cudaMalloc(&data, vectorsize * sizeof(float)));

    for (round = 0; round < num_recurrence; round++)  {
        /* TODO: type, what happens if LSTM is not the input layer? */
        const float alpha = 1;
        const float beta = 1;
        int incx = 1;
        int incy = 1;

        cuda_check_error(cudaMemcpy(z_i, state->b_i[layer], vectorsize * cell_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        cuda_check_error(cudaMemcpy(z_f, state->b_f[layer], vectorsize * cell_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        cuda_check_error(cudaMemcpy(z_c, state->b_c[layer], vectorsize * cell_dim * sizeof(float), cudaMemcpyDeviceToDevice));
        cuda_check_error(cudaMemcpy(z_o, state->b_o[layer], vectorsize * cell_dim * sizeof(float), cudaMemcpyDeviceToDevice));

        cuda_check_error(cudaMemcpy(data, (float*)op->data[state->arg_col_map[round]], vectorsize * sizeof(float), cudaMemcpyHostToDevice));

        cublas_check_error(cublasSger(handle, cell_dim, vectorsize, &alpha, state->W_i[layer], incx, data, incy, z_i, cell_dim));
        cublas_check_error(cublasSger(handle, cell_dim, vectorsize, &alpha, state->W_f[layer], incx, data, incy, z_f, cell_dim));
        cublas_check_error(cublasSger(handle, cell_dim, vectorsize, &alpha, state->W_c[layer], incx, data, incy, z_c, cell_dim));
        cublas_check_error(cublasSger(handle, cell_dim, vectorsize, &alpha, state->W_o[layer], incx, data, incy, z_o, cell_dim));

        if (h_tm1) {
            cublas_check_error(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cell_dim, vectorsize, cell_dim, &alpha, state->U_i[layer], cell_dim, h_tm1, cell_dim, &beta, z_i, cell_dim));
            cublas_check_error(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cell_dim, vectorsize, cell_dim, &alpha, state->U_f[layer], cell_dim, h_tm1, cell_dim, &beta, z_f, cell_dim));
            cublas_check_error(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cell_dim, vectorsize, cell_dim, &alpha, state->U_c[layer], cell_dim, h_tm1, cell_dim, &beta, z_c, cell_dim));
            cublas_check_error(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cell_dim, vectorsize, cell_dim, &alpha, state->U_o[layer], cell_dim, h_tm1, cell_dim, &beta, z_o, cell_dim));
        }

        // begin _compute_carry_and_output_fused

        vector_activate_cuda(z_i, vectorsize * cell_dim, SIGMOID);
        vector_activate_cuda(z_f, vectorsize * cell_dim, SIGMOID);
        vector_activate_cuda(z_c, vectorsize * cell_dim, TANH);        
        vector_elementwise_mul_cuda(z_i, z_c, z_c, cell_dim * vectorsize);

        if (c_tm1) {
            vector_elementwise_mul_cuda(z_f, c_tm1, c_tm1, vectorsize * cell_dim);
            vector_elementwise_add_cuda(z_c, c_tm1, c_tm1, vectorsize * cell_dim);
        } else {
            cuda_check_error(cudaMalloc(&c_tm1, vectorsize * cell_dim * sizeof(float)));
            cuda_check_error(cudaMemcpy(c_tm1, z_c, vectorsize * cell_dim * sizeof(float), cudaMemcpyHostToDevice));
        }
        vector_activate_cuda(z_o, vectorsize * cell_dim, SIGMOID);

        // end _compute_carry_and_output_fused

        if (!h_tm1) cuda_check_error(cudaMalloc(&h_tm1, vectorsize * cell_dim * sizeof(float)));
        cuda_check_error(cudaMemcpy(h_tm1, c_tm1, vectorsize * cell_dim * sizeof(float), cudaMemcpyHostToDevice));
        vector_activate_cuda(h_tm1, vectorsize * cell_dim, TANH);
        vector_elementwise_mul_cuda(z_o, h_tm1, h_tm1, vectorsize * cell_dim);
    }

    cuda_check_error(cudaFree(z_i));
    cuda_check_error(cudaFree(z_f));
    cuda_check_error(cudaFree(z_c));
    cuda_check_error(cudaFree(z_o));
    if (c_tm1) cuda_check_error(cudaFree(c_tm1));

    *int_rows = state->layer_dims[layer];
    *int_cols = vectorsize;
    return h_tm1;
}