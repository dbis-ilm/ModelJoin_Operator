#include <string.h>
#include <stdio.h>
#include "mj_common.h"

void print_matrix(float *mat, int rows, int cols)
{
    int i, k;
    for (i = 0; i < rows; i++) {
        printf("(");
        for (k = 0; k < cols; k++) {
            printf("%f, ", mat[i + k * rows]);
        }
        printf(")\n");
    }
}

static bool model_has_lstm_layer(int num_layers, TFLayerType *layer_types) {
    int i;
    for (i = 0; i < num_layers; i++) {
        if (layer_types[i] == LSTM) return true;
    }
    return false;
}

#ifndef USECUDA
#include <mkl.h>

void allocate_weights(ModeljoinState *state, int num_in_cols, 
    int num_layers,  int *layer_dims, TFLayerType *layer_types)
{
    int i;
    int alignment = 32;

    state->W_i = calloc(num_layers, sizeof(float*));
    state->b_i = calloc(num_layers, sizeof(float*));

    if (model_has_lstm_layer(num_layers, layer_types)) {
        state->W_f = calloc(num_layers, sizeof(float*));
        state->W_c = calloc(num_layers, sizeof(float*));
        state->W_o = calloc(num_layers, sizeof(float*));
        state->U_i = calloc(num_layers, sizeof(float*));
        state->U_f = calloc(num_layers, sizeof(float*));
        state->U_c = calloc(num_layers, sizeof(float*));
        state->U_o = calloc(num_layers, sizeof(float*));
        state->b_f = calloc(num_layers, sizeof(float*));
        state->b_c = calloc(num_layers, sizeof(float*));
        state->b_o = calloc(num_layers, sizeof(float*));
    }
    for (i = 0; i < num_layers; i++) {
        if (layer_types[i] == LSTM) {
            state->W_i[i] = (float*) mkl_malloc(layer_dims[i] * sizeof(float), alignment);
            state->W_f[i] = (float*) mkl_malloc(layer_dims[i] * sizeof(float), alignment);
            state->W_c[i] = (float*) mkl_malloc(layer_dims[i] * sizeof(float), alignment);
            state->W_o[i] = (float*) mkl_malloc(layer_dims[i] * sizeof(float), alignment);
            state->U_i[i] = (float*) mkl_malloc(layer_dims[i] * layer_dims[i] * sizeof(float), alignment);
            state->U_f[i] = (float*) mkl_malloc(layer_dims[i] * layer_dims[i] * sizeof(float), alignment);
            state->U_c[i] = (float*) mkl_malloc(layer_dims[i] * layer_dims[i] * sizeof(float), alignment);
            state->U_o[i] = (float*) mkl_malloc(layer_dims[i] * layer_dims[i] * sizeof(float), alignment);
            state->b_i[i] = (float*) mkl_malloc(layer_dims[i] * sizeof(float), alignment);
            state->b_f[i] = (float*) mkl_malloc(layer_dims[i] * sizeof(float), alignment);
            state->b_c[i] = (float*) mkl_malloc(layer_dims[i] * sizeof(float), alignment);
            state->b_o[i] = (float*) mkl_malloc(layer_dims[i] * sizeof(float), alignment);
        } else {
            int rows = i == 0 ? num_in_cols : layer_dims[i - 1];
            state->W_i[i] = (float*) mkl_malloc(rows * layer_dims[i] * sizeof(float), alignment);
            state->b_i[i] = (float*) mkl_malloc(layer_dims[i] * sizeof(float), alignment);
        }
    }
}

void free_weights(ModeljoinState *state) {
    int i;
    for (i = 0; i < state->num_layers; i++) {
        mkl_free(state->W_i[i]);
        mkl_free(state->b_i[i]);

        if (state->W_f && state->W_f[i]) mkl_free(state->W_f[i]);
        if (state->W_c && state->W_c[i]) mkl_free(state->W_c[i]);
        if (state->W_o && state->W_o[i]) mkl_free(state->W_o[i]);
        if (state->U_i && state->U_i[i]) mkl_free(state->U_i[i]);
        if (state->U_f && state->U_f[i]) mkl_free(state->U_f[i]);
        if (state->U_c && state->U_c[i]) mkl_free(state->U_c[i]);
        if (state->U_o && state->U_o[i]) mkl_free(state->U_o[i]);
        if (state->b_f && state->b_f[i]) mkl_free(state->b_f[i]);
        if (state->b_c && state->b_c[i]) mkl_free(state->b_c[i]);
        if (state->b_o && state->b_o[i]) mkl_free(state->b_o[i]);
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

#endif // USECUDA

void mj_result_to_buffer(float *buf_col, float *result, int vectorsize)
{
    memcpy(buf_col, result, vectorsize * sizeof(float));
}