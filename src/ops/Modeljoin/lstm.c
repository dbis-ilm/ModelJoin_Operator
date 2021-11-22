#ifndef USECUDA

#include <mkl.h>
#include "mj_common.h"
#include "../../operator.h"
#include "activation.h"
#include "lstm.h"

void lstm_layer_consume_tuple(ModeljoinState *state, int logical_layer, Buffer data, int row) 
{
    long node_in = ((long*)data[Node_in_off])[row];
    long node = ((long*)data[Node_off])[row];

    /* TODO: what if weight is really zero*/
    if (((float*)data[W_i_off])[row] != 0) {
        long mat_position = node;
        assert(node == node_in);
        state->W_i[logical_layer][mat_position] =
            ((float*)data[W_i_off])[row];
        state->W_f[logical_layer][mat_position] =
            ((float*)data[W_f_off])[row];
        state->W_c[logical_layer][mat_position] =
            ((float*)data[W_c_off])[row];
        state->W_o[logical_layer][mat_position] =
            ((float*)data[W_o_off])[row];
        state->b_i[logical_layer][node] =
            ((float*)data[b_i_off])[row];
        state->b_f[logical_layer][node] =
            ((float*)data[b_f_off])[row];
        state->b_c[logical_layer][node] =
            ((float*)data[b_c_off])[row];
        state->b_o[logical_layer][node] =
            ((float*)data[b_o_off])[row];
    } else {
        /* We later want to compute x*A, but Intel MKL only offers A*x. 
        Therefore we save the transposed matrix to compute x*A = A_t * x_t 
        Consequently, the element (node_in, node) is saved at (node, node_in), 
        which translates to node + node_in * dim */

        long mat_position = node + node_in * state->layer_dims[logical_layer];
        state->U_i[logical_layer][mat_position] =
            ((float*)data[U_i_off])[row];
        state->U_f[logical_layer][mat_position] =
            ((float*)data[U_f_off])[row];
        state->U_c[logical_layer][mat_position] =
            ((float*)data[U_c_off])[row];
        state->U_o[logical_layer][mat_position] =
            ((float*)data[U_o_off])[row];
    }
}

void lstm_layer_finish(void *o, int layer) 
{
    Operator *op = (Operator*)o;
    ModeljoinState *state = (ModeljoinState*) op->state;
    int cell_dim = state->layer_dims[layer];
    int i;
    int alignment = 32;

    float *new_b_i, *new_b_f, *new_b_c, *new_b_o;
    new_b_i = mkl_malloc(cell_dim * op->vectorsize * sizeof(float), alignment);
    new_b_f = mkl_malloc(cell_dim * op->vectorsize * sizeof(float), alignment);
    new_b_c = mkl_malloc(cell_dim * op->vectorsize * sizeof(float), alignment);
    new_b_o = mkl_malloc(cell_dim * op->vectorsize * sizeof(float), alignment);

    for (i = 0; i < op->vectorsize; i++) {
        memcpy(&(new_b_i[i * cell_dim]), state->b_i[layer], cell_dim * sizeof(float));
        memcpy(&(new_b_f[i * cell_dim]), state->b_f[layer], cell_dim * sizeof(float));
        memcpy(&(new_b_c[i * cell_dim]), state->b_c[layer], cell_dim * sizeof(float));
        memcpy(&(new_b_o[i * cell_dim]), state->b_o[layer], cell_dim * sizeof(float));
    }

    mkl_free(state->b_i[layer]);
    mkl_free(state->b_f[layer]);
    mkl_free(state->b_c[layer]);
    mkl_free(state->b_o[layer]);

    state->b_i[layer] = new_b_i;
    state->b_f[layer] = new_b_f;
    state->b_c[layer] = new_b_c;
    state->b_o[layer] = new_b_o;
}

float* lstm_layer_forward(void *o, int layer, float *intermediate, int *int_rows, int *int_cols)
{
    Operator *op = (Operator*)o;
    ModeljoinState *state = (ModeljoinState*) op->state;

    /* TODO: Rough assumption here */
    int num_recurrence = state->num_in_cols;
    int round;

    /* Calculation based on https://github.com/keras-team/keras/blob/master/keras/layers/recurrent.py LSTMCell:call() */
    
    MKL_INT cell_dim = state->layer_dims[layer];
    MKL_INT vectorsize = op->vectorsize;
    float *h_tm1 = NULL; /* Memory State, vector of size cell_dim*/
    float *c_tm1 = NULL; /* Cell State, vector size cell_dim */
    
    float *z_i = calloc(vectorsize * cell_dim, sizeof(float));
    float *z_f = calloc(vectorsize * cell_dim, sizeof(float));
    float *z_c = calloc(vectorsize * cell_dim, sizeof(float));
    float *z_o = calloc(vectorsize * cell_dim, sizeof(float));

    for (round = 0; round < num_recurrence; round++)  {
        float *data = (float*)op->data[state->arg_col_map[round]];
        const char transa = 'N';
        const char transb = 'N';
        const float alpha = 1;
        const float beta = 1;
        MKL_INT incx = 1;
        MKL_INT incy = 1;

        memcpy(z_i, state->b_i[layer], vectorsize * cell_dim * sizeof(float));
        memcpy(z_f, state->b_f[layer], vectorsize * cell_dim * sizeof(float));
        memcpy(z_c, state->b_c[layer], vectorsize * cell_dim * sizeof(float));
        memcpy(z_o, state->b_o[layer], vectorsize * cell_dim * sizeof(float));

        sger(&cell_dim, &vectorsize, &alpha, state->W_i[layer], &incx, data, &incy, z_i, &cell_dim);
        sger(&cell_dim, &vectorsize, &alpha, state->W_f[layer], &incx, data, &incy, z_f, &cell_dim);
        sger(&cell_dim, &vectorsize, &alpha, state->W_c[layer], &incx, data, &incy, z_c, &cell_dim);
        sger(&cell_dim, &vectorsize, &alpha, state->W_o[layer], &incx, data, &incy, z_o, &cell_dim);

        if (h_tm1) {
            sgemm(&transa, &transb, &cell_dim, &vectorsize, &cell_dim, &alpha, state->U_i[layer], &cell_dim, h_tm1, &cell_dim, &beta, z_i, &cell_dim);
            sgemm(&transa, &transb, &cell_dim, &vectorsize, &cell_dim, &alpha, state->U_f[layer], &cell_dim, h_tm1, &cell_dim, &beta, z_f, &cell_dim);
            sgemm(&transa, &transb, &cell_dim, &vectorsize, &cell_dim, &alpha, state->U_c[layer], &cell_dim, h_tm1, &cell_dim, &beta, z_c, &cell_dim);
            sgemm(&transa, &transb, &cell_dim, &vectorsize, &cell_dim, &alpha, state->U_o[layer], &cell_dim, h_tm1, &cell_dim, &beta, z_o, &cell_dim);
        }

        // begin _compute_carry_and_output_fused

        vector_activate(z_i, vectorsize * cell_dim, SIGMOID);
        vector_activate(z_f, vectorsize * cell_dim, SIGMOID);
        vector_activate(z_c, vectorsize * cell_dim, TANH);        
        vsMul(cell_dim * vectorsize, z_i, z_c, z_c);

        if (c_tm1) {
            vsMul(vectorsize * cell_dim, z_f, c_tm1, c_tm1);
            vsAdd(vectorsize     * cell_dim, z_c, c_tm1, c_tm1);
        } else {
            c_tm1 = calloc(vectorsize * cell_dim, sizeof(float));
            memcpy(c_tm1, z_c, vectorsize * cell_dim * sizeof(float));
        }
        vector_activate(z_o, vectorsize * cell_dim, SIGMOID);

        // end _compute_carry_and_output_fused

        if (!h_tm1) h_tm1 = calloc(vectorsize * cell_dim, sizeof(float));
        memcpy(h_tm1, c_tm1, vectorsize * cell_dim * sizeof(float));
        vector_activate(h_tm1, vectorsize * cell_dim, TANH);
        vsMul(vectorsize * cell_dim, z_o, h_tm1, h_tm1);
    }

    free(z_i);
    free(z_f);
    free(z_c);
    free(z_o);
    if (c_tm1) free(c_tm1);
    
    *int_rows = cell_dim;
    *int_cols = vectorsize;
    return h_tm1;
}

#endif // USECUDA