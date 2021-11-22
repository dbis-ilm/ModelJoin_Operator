#ifndef USECUDA

#include <mkl.h>
#include "mj_common.h"
#include "../../operator.h"

#include "dense.h"

void dense_layer_consume_tuple(ModeljoinState *state, int logical_layer, Buffer data, int row) 
{
    long node_in = ((long*)data[Node_in_off])[row];
    long node = ((long*)data[Node_off])[row];
    
    /* We later want to compute x*A, but Intel MKL only offers A*x. 
    Therefore we save the transposed matrix to compute x*A = A_t * x_t 
    Consequently, the element (node_in, node) is saved at (node, node_in), 
    which translates to node + node_in * dim */
    state->W_i[logical_layer][node + node_in * state->layer_dims[logical_layer]] =
        ((float*)data[W_i_off])[row];
    state->b_i[logical_layer][node] =
        ((float*)data[b_i_off])[row];
}

void dense_layer_finish(void *o, int layer) 
{
    Operator *op = (Operator*)o;
    ModeljoinState *state = (ModeljoinState*) op->state;
    int cell_dim = state->layer_dims[layer];
    int i;
    int alignment = 32;

    float *new_bias;
    new_bias = mkl_malloc(cell_dim * op->vectorsize * sizeof(float), alignment);

    for (i = 0; i < op->vectorsize; i++) {
       memcpy(&(new_bias[i * cell_dim]), state->b_i[layer], cell_dim * sizeof(float));
    }

    mkl_free(state->b_i[layer]);

    state->b_i[layer] = new_bias;
}

float* dense_layer_forward_rowwise(void *o, int layer, float *intermediate, int *int_rows, int *int_cols)
{
    Operator *op = (Operator*)o;
    ModeljoinState *state = (ModeljoinState*) op->state;
    const char trans = 'N';
    /* cols of the transposed matrix is equal to the vector length */
    MKL_INT cols = intermediate ? *int_rows : state->num_in_cols;
    /* rows is the layer's dimension and determines the output vector size of the layer */
    MKL_INT rows = state->layer_dims[layer];
    const float alpha = 1;
    const float beta = 1;
    MKL_INT incx = 1;
    MKL_INT incy = 1;
    float *mat = state->W_i[layer];
    float *bias = state->b_i[layer];
    float *x;
    float *result = calloc(state->layer_dims[layer] * op->vectorsize, sizeof(float));
    int i;
    int row;
    bool allocated = false;

    memcpy(result, bias, op->vectorsize * state->layer_dims[layer] * sizeof(float));
    
    for (row = 0; row < op->vectorsize; row++) {
        if (!intermediate) {
                /* Input layer */
                x = calloc(cols, sizeof(float));
                allocated = true;
                /* TODO: type */
                for (i = 0; i < cols; i++) x[i] = ((float*)op->data[state->arg_col_map[i]])[row];
            } else {
                x = &(intermediate[row * state->layer_dims[layer -1]]);
            }
            
            sgemv(&trans, &rows, &cols, &alpha, mat, &rows, x, &incx, &beta, &(result[row * rows]), &incy);
            if (allocated) free(x);
    }

    *int_rows = state->layer_dims[layer];
    *int_cols = op->vectorsize;
    return result;
}


float* dense_layer_forward_matrix_manual_loading(void *o, int layer, float *intermediate, int *int_rows, int *int_cols)
{
    Operator *op = (Operator*)o;
    ModeljoinState *state = (ModeljoinState*) op->state;
    const char transa = 'N';
    const char transb = 'N';
    /* cols of the transposed matrix is equal to the vector length */
    MKL_INT cols = intermediate ? *int_rows : state->num_in_cols;
    /* rows is the layer's dimension and determines the output vector size of the layer */
    MKL_INT rows = state->layer_dims[layer];
    MKL_INT vectorsize = op->vectorsize;
    const float alpha = 1;
    const float beta = 1;
    float *mat = state->W_i[layer];
    float *bias = state->b_i[layer];
    float *x;
    float *result = calloc(rows * op->vectorsize, sizeof(float));
    int i;
    bool allocated = false;

    if (!intermediate) {
        int k;
        /* Input layer */
        x = calloc(cols * vectorsize, sizeof(float));
        allocated = true;

        for (k = 0; k < op->vectorsize; k++) {
            for (i = 0; i < cols; i++) {
                x[i + k * cols] = ((float*)op->data[state->arg_col_map[i]])[k];
            }
        }
    } else {
        x = intermediate;
    }

    memcpy(result, bias, vectorsize * state->layer_dims[layer] * sizeof(float));
    sgemm(&transa, &transb, &rows, &vectorsize, &cols, &alpha, mat, &rows, x, &cols, &beta, result, &rows);

    if (allocated) free(x);

    *int_rows = state->layer_dims[layer];
    *int_cols = vectorsize;
    return result;
}

float* dense_layer_forward_matrix_memcpy_loading(void *o, int layer, float *intermediate, int *int_rows, int *int_cols)
{
    Operator *op = (Operator*)o;
    ModeljoinState *state = (ModeljoinState*) op->state;
    const char transa = 'N';
    const char transb = 'N';
    /* cols of the transposed matrix is equal to the vector length */
    MKL_INT cols = intermediate ? *int_rows : state->num_in_cols;
    /* rows is the layer's dimension and determines the output vector size of the layer */
    MKL_INT rows = state->layer_dims[layer];
    MKL_INT vectorsize = op->vectorsize;
    const float alpha = 1;
    const float beta = 1;
    float *mat = state->W_i[layer];
    float *bias = state->b_i[layer];
    float *x;
    float *result = calloc(rows * op->vectorsize, sizeof(float));
    int i;
    bool allocated = false;

    if (!intermediate) {
        const char ordering = 'R';
        const char trans = 'T';
        /* Input layer */
        x = calloc(cols * vectorsize, sizeof(float));
        allocated = true;
        /* TODO: type */
        for (i = 0; i < cols; i++) {
            memcpy(&x[i * vectorsize], ((float*)op->data[state->arg_col_map[i]]), vectorsize * sizeof(float));   
        }
        /* Transpose */
        mkl_simatcopy(ordering, trans, cols, vectorsize, alpha, x, vectorsize, cols);
    } else {
        x = intermediate;
    }
    memcpy(result, bias, vectorsize * state->layer_dims[layer] * sizeof(float));
    sgemm(&transa, &transb, &rows, &vectorsize, &cols, &alpha, mat, &rows, x, &cols, &beta, result, &rows);

    if (allocated) free(x);

    *int_rows = state->layer_dims[layer];
    *int_cols = vectorsize;
    return result;
}

#endif // USECUDA