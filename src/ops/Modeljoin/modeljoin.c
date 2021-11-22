#include <stdbool.h>
#include "modeljoin.h"
#include "mj_common.h"
#include "dense.h"
#include "lstm.h"
#include "activation.h"

void modeljoin_close(Operator *op) {
    ModeljoinState *state = (ModeljoinState*) op->state;
#ifdef USECUDA
    free_weights_cuda(state);
#else
    free_weights(state);
#endif
    free(state);
    operator_free(op);
}

/* In the SQL table, logical LSTM layers are splitted into two physical layers. The modeljoin operator however
    does not need this, it maintains the set of matrices in a single layer index. Therefore we compute a mapping
    between physical and logical mapping. For example, take a model with {LSTM, DENSE, DENSE}, which is represented
    in four physical layers. The mapping would be [0,0,1,2], which means that the first two physical layers map to the 
    first logical level */
static int *physical_to_logical_layer_mapping(int num_layers, TFLayerType *layer_types)
{
    int i;
    int num_physical_layers = num_layers;
    int *mapping = NULL;
    int cur_layer = 0;

    for (i = 0; i < num_layers; i++) {
        if(layer_types[i] == LSTM) num_physical_layers++;
    }

    mapping = calloc(num_physical_layers, sizeof(int));

    for (i = 0; i < num_layers; i++) {
        if(layer_types[i] == LSTM) {
            mapping[cur_layer] = i;
            mapping[cur_layer + 1] = i;
            cur_layer += 2;
        } else if (layer_types[i] == DENSE) {
            mapping[cur_layer] = i;
            cur_layer++;
        } else {
            assert(false);
        }
    }
    return mapping;
}

typedef void (*layer_consume_tuple_fcn)(ModeljoinState *state, int logical_layer, Buffer data, int row);

static layer_consume_tuple_fcn *layer_consumption_functions(ModeljoinState *state)
{
    layer_consume_tuple_fcn *functions = calloc(state->num_layers, sizeof(layer_consume_tuple_fcn));
    int i;

#ifdef USECUDA
    for (i = 0; i < state->num_layers; i++) {
        if (state->layer_types[i] == DENSE) {
            functions[i] = &dense_layer_consume_tuple_cuda;
        } else if (state->layer_types[i] == LSTM) {
            functions[i] = &lstm_layer_consume_tuple_cuda;
        } else {
            assert(false);
        }
    }
#else
    for (i = 0; i < state->num_layers; i++) {
        if (state->layer_types[i] == DENSE) {
            functions[i] = &dense_layer_consume_tuple;
        } else if (state->layer_types[i] == LSTM) {
            functions[i] = &lstm_layer_consume_tuple;
        } else {
            assert(false);
        }
    }
#endif
    return functions;
}

static layer_forward_fcn *layer_forward_functions(ModeljoinState *state)
{
    layer_forward_fcn *functions = calloc(state->num_layers, sizeof(layer_forward_fcn));
    int i;

#ifdef USECUDA
    for (i = 0; i < state->num_layers; i++) {
        if (state->layer_types[i] == DENSE) {
            functions[i] = &dense_layer_forward_matrix_memcpy_loading_cuda;
        } else if (state->layer_types[i] == LSTM) {
            functions[i] = &lstm_layer_forward_cuda;
        } else {
            assert(false);
        }
    }
#else
    for (i = 0; i < state->num_layers; i++) {
        if (state->layer_types[i] == DENSE) {
            functions[i] = &dense_layer_forward_matrix_memcpy_loading;
        } else if (state->layer_types[i] == LSTM) {
            functions[i] = &lstm_layer_forward;
        } else {
            assert(false);
        }
    }
#endif
    return functions;
}

static model_build_finish_fcn *model_build_finish_functions(ModeljoinState *state)
{
    model_build_finish_fcn *functions = calloc(state->num_layers, sizeof(model_build_finish_fcn));
    int i;

#ifdef USECUDA
    for (i = 0; i < state->num_layers; i++) {
        if (state->layer_types[i] == DENSE) {
            functions[i] = &dense_layer_finish_cuda;
        } else if (state->layer_types[i] == LSTM) {
            functions[i] = &lstm_layer_finish_cuda;
        } else {
            assert(false);
        }
    }
#else
    for (i = 0; i < state->num_layers; i++) {
        if (state->layer_types[i] == DENSE) {
            functions[i] = &dense_layer_finish;
        } else if (state->layer_types[i] == LSTM) {
            functions[i] = &lstm_layer_finish;
        } else {
            assert(false);
        }
    }
#endif
    return functions;
}

static void consume_and_build_model(Operator *op)
{
    ModeljoinState *state = (ModeljoinState*) op->state;
    int *physical_logical_mapping = physical_to_logical_layer_mapping(state->num_layers, state->layer_types);
    layer_consume_tuple_fcn *lcf = layer_consumption_functions(state);
    int i;

    /* Get vector of model table */
    int n = op->rchild->next(op->rchild);

    while (n > 0) {
        Buffer data = op->rchild->data;
#ifdef PROFILE
    profile_start(op->profile_stats);
#endif      
        for (i = 0; i < n; i++) {
            long physical_layer = ((long*)data[Layer_in_off])[i];
            long logical_layer;

            /* Skip the artificial input transposition layer */
            if (physical_layer == -1) continue;
            logical_layer = physical_logical_mapping[physical_layer];
            lcf[logical_layer](state, logical_layer, data, i);
        }
#ifdef PROFILE
    profile_stop(op->profile_stats, EXECUTE);
#endif 
        n = op->rchild->next(op->rchild);
    }


#ifdef PROFILE
    profile_start(op->profile_stats);
#endif    
    for (i = 0; i < state->num_layers; i++) state->model_build_finish[i](op, i);
    free(physical_logical_mapping);
    free(lcf);
#ifdef PROFILE
    profile_stop(op->profile_stats, EXECUTE);
#endif 
    state->built = true;
}

static int modeljoin_next(Operator *op) 
{
    int added_col_idx;
    int i;
    ModeljoinState *state = (ModeljoinState*) op->state;
    float *intermediate = NULL;
    int int_rows = 0;
    int int_cols = 0;
    int n;

    if (!state->built) consume_and_build_model(op);

    n = op->lchild->next(op->lchild);
#ifdef PROFILE
    profile_start(op->profile_stats);
#endif
    op->data = op->lchild->data;
    op->data = reallocate_buffer(op->data, op->lchild->num_cols, op->num_cols, op->col_types, op->vectorsize);
    added_col_idx = op->num_cols - 1;

    // We can return only after reallocation, otherwise the buffer does not match op->num_cols
    if (n == 0) goto end;

    for (i = 0; i < state->num_layers; i++) {
        float *old = intermediate;
        intermediate = state->layer_forward[i](op, i, intermediate, &int_rows, &int_cols);

#ifdef USECUDA
        vector_activate_cuda_unmangled(intermediate, int_rows * int_cols, state->layer_activations[i]);
        if (old) cudaFree(old);
#else
        vector_activate(intermediate, int_rows * int_cols, state->layer_activations[i]);
        if (old) free(old);
#endif
    }
    
    assert(int_rows == 1);
    assert(int_cols == op->vectorsize);
#ifdef USECUDA
    mj_result_to_buffer_cuda((float*)op->data[added_col_idx], intermediate, op->vectorsize);
    if (intermediate) cudaFree(intermediate);
#else
    mj_result_to_buffer((float*)op->data[added_col_idx], intermediate, op->vectorsize);
    if (intermediate) free(intermediate);
#endif
    
end:
#ifdef PROFILE
    profile_stop(op->profile_stats, EXECUTE);
#endif
    return n;
}

Operator *modeljoin_build(Operator *model, Operator *child, int num_in_cols, int *arg_col_map,
    int num_layers, int *layer_dims, TFLayerType *layer_types, TFActivationFunction *layer_activations)
{
    Operator *op;
    ModeljoinState *state;

    /* We expect a specific model table schema */
    assert(model->num_cols == 16);
#ifdef USECUDA
    op = operator_alloc(&modeljoin_next, &modeljoin_close, child, model, child->num_cols + 1, child->col_types, "Modeljoin(CUDA)");
#else
    op = operator_alloc(&modeljoin_next, &modeljoin_close, child, model, child->num_cols + 1, child->col_types, "Modeljoin(Intel MKL)");
#endif
    /* Prediction is an added column 
        TODO: generic*/
    op->col_types[op->num_cols - 1] = FLOAT;
    op->state = calloc(1, sizeof(ModeljoinState));
    state = ((ModeljoinState*)(op->state));
    state->built = false;
    state->num_layers = num_layers;
    state->num_in_cols = num_in_cols;

    state->layer_dims = layer_dims;
    state->layer_types = layer_types;
    state->layer_activations = layer_activations;
    state->arg_col_map = arg_col_map;

    state->layer_forward = layer_forward_functions(state);
    state->model_build_finish = model_build_finish_functions(state);

#ifdef USECUDA
    allocate_weights_cuda(state, num_in_cols, num_layers, layer_dims, layer_types);
#else
    allocate_weights(state, num_in_cols, num_layers, layer_dims, layer_types);
#endif

#ifdef PROFILE
    profile_stop(op->profile_stats, BUILD);
#endif

    return op;
}