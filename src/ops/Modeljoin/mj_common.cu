#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "mj_common.h"
#include "../../utils/cuda_utils.h"

static bool model_has_lstm_layer(int num_layers, TFLayerType *layer_types) {
    int i;
    for (i = 0; i < num_layers; i++) {
        if (layer_types[i] == LSTM) return true;
    }
    return false;
}

extern "C" void allocate_weights_cuda(ModeljoinState *state, int num_in_cols, 
    int num_layers,  int *layer_dims, TFLayerType *layer_types)
{
    int i;

    state->W_i = (float**) calloc(num_layers, sizeof(float*));
    state->b_i = (float**) calloc(num_layers, sizeof(float*));

    if (model_has_lstm_layer(num_layers, layer_types)) {
        state->W_f = (float**) calloc(num_layers, sizeof(float*));
        state->W_c = (float**) calloc(num_layers, sizeof(float*));
        state->W_o = (float**) calloc(num_layers, sizeof(float*));
        state->U_i = (float**) calloc(num_layers, sizeof(float*));
        state->U_f = (float**) calloc(num_layers, sizeof(float*));
        state->U_c = (float**) calloc(num_layers, sizeof(float*));
        state->U_o = (float**) calloc(num_layers, sizeof(float*));
        state->b_f = (float**) calloc(num_layers, sizeof(float*));
        state->b_c = (float**) calloc(num_layers, sizeof(float*));
        state->b_o = (float**) calloc(num_layers, sizeof(float*));
    }
    for (i = 0; i < num_layers; i++) {
        if (layer_types[i] == LSTM) {
            cuda_check_error(cudaMalloc(&(state->W_i[i]), layer_dims[i] * sizeof(float)));
            cuda_check_error(cudaMalloc(&(state->W_f[i]), layer_dims[i] * sizeof(float)));
            cuda_check_error(cudaMalloc(&(state->W_c[i]), layer_dims[i] * sizeof(float)));
            cuda_check_error(cudaMalloc(&(state->W_o[i]), layer_dims[i] * sizeof(float)));
            cuda_check_error(cudaMalloc(&(state->U_i[i]), layer_dims[i] * layer_dims[i] * sizeof(float)));
            cuda_check_error(cudaMalloc(&(state->U_f[i]), layer_dims[i] * layer_dims[i] * sizeof(float)));
            cuda_check_error(cudaMalloc(&(state->U_c[i]), layer_dims[i] * layer_dims[i] * sizeof(float)));
            cuda_check_error(cudaMalloc(&(state->U_o[i]), layer_dims[i] * layer_dims[i] * sizeof(float)));
            cuda_check_error(cudaMalloc(&(state->b_i[i]), layer_dims[i] * sizeof(float)));
            cuda_check_error(cudaMalloc(&(state->b_f[i]), layer_dims[i] * sizeof(float)));
            cuda_check_error(cudaMalloc(&(state->b_c[i]), layer_dims[i] * sizeof(float)));
            cuda_check_error(cudaMalloc(&(state->b_o[i]), layer_dims[i] * sizeof(float)));
        } else {
            int rows = i == 0 ? num_in_cols : layer_dims[i - 1];
            cuda_check_error(cudaMalloc(&(state->W_i[i]), rows * layer_dims[i] * sizeof(float)));
            cuda_check_error(cudaMalloc(&(state->b_i[i]), layer_dims[i] * sizeof(float)));
        }
    }
}

extern "C" void free_weights_cuda(ModeljoinState *state)
{
    int i;
    for (i = 0; i < state->num_layers; i++) {
        cuda_check_error(cudaFree(state->W_i[i]));
        cuda_check_error(cudaFree(state->b_i[i]));

        if (state->W_f && state->W_f[i]) cuda_check_error(cudaFree(state->W_f[i]));
        if (state->W_c && state->W_c[i]) cuda_check_error(cudaFree(state->W_c[i]));
        if (state->W_o && state->W_o[i]) cuda_check_error(cudaFree(state->W_o[i]));
        if (state->U_i && state->U_i[i]) cuda_check_error(cudaFree(state->U_i[i]));
        if (state->U_f && state->U_f[i]) cuda_check_error(cudaFree(state->U_f[i]));
        if (state->U_c && state->U_c[i]) cuda_check_error(cudaFree(state->U_c[i]));
        if (state->U_o && state->U_o[i]) cuda_check_error(cudaFree(state->U_o[i]));
        if (state->b_f && state->b_f[i]) cuda_check_error(cudaFree(state->b_f[i]));
        if (state->b_c && state->b_c[i]) cuda_check_error(cudaFree(state->b_c[i]));
        if (state->b_o && state->b_o[i]) cuda_check_error(cudaFree(state->b_o[i]));
    }

    free(state->W_i);
    free(state->b_i);

    if (state->W_f) free(state->W_f);
    if (state->W_c) free(state->W_c);
    if (state->W_o) free(state->W_o);
    if (state->U_i) free(state->U_i);
    if (state->U_f) free(state->U_f);
    if (state->U_c) free(state->U_c);
    if (state->U_o) free(state->U_o);
    if (state->b_f) free(state->b_f);
    if (state->b_c) free(state->b_c);
    if (state->b_o) free(state->b_o);
}

extern "C" void mj_result_to_buffer_cuda(float *buf_col, float *result, int vectorsize)
{
    cuda_check_error(cudaMemcpy(buf_col, result, vectorsize * sizeof(float), cudaMemcpyDeviceToHost));
}