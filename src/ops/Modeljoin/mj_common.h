#ifndef ML_COMMON_H
#define ML_COMMON_H

#include <stdbool.h>

typedef enum TFLayerType_t { 
    DENSE = 1,
    LSTM = 2
} TFLayerType;

typedef enum TFActivationFunction_t { 
    LINEAR = 1,
    RELU = 2,
    SIGMOID = 3,
    TANH = 4
} TFActivationFunction;

/* Column offsets in the model table */
#define Layer_in_off 0
#define Node_in_off 1
#define Layer_off 2
#define Node_off 3
#define W_i_off 4
#define W_f_off 5
#define W_c_off 6
#define W_o_off 7
#define U_i_off 8
#define U_f_off 9
#define U_c_off 10
#define U_o_off 11
#define b_i_off 12
#define b_f_off 13
#define b_c_off 14
#define b_o_off 15

/** Function definition of any layer forward function, transforming an potential empty intermediate result from 
 * a preceeding layer into the layer output using matrix operations. */
typedef float* (*layer_forward_fcn)(void *op, int layer, float *intermediate, int *int_rows, int *int_cols);

/** After model build phase finished, a finish function is called on each layer, to clean up or manipulate once.*/
typedef void (*model_build_finish_fcn)(void *op, int layer);

/** The state of the modeljoin operator */
typedef struct ModeljoinState_t {
    /** Number of model inputs */
    int num_in_cols;
    /** Offsets of the input columns in the input buffer */
    int *arg_col_map;    
    /** Number of layers of the model */
    int num_layers;
    /** Dimensions of the model layers */
    int *layer_dims;
    /** Layer types */
    TFLayerType *layer_types;
    /** Layer activation functions */
    TFActivationFunction *layer_activations;
    /** Indicate whether build phase already finished*/
    bool built;
    /** Kernel for input gate per layer */
    float **W_i;
    /** Kernel for forget gate per layer */
    float **W_f;
    /** Kernel for cell gate per layer */
    float **W_c;
    /** Kernel for output gate per layer */
    float **W_o;
    /** Recurrent kernel for input gate per layer */
    float **U_i;
    /** Recurrent kernel for forget gate per layer */
    float **U_f;
    /** Recurrent kernel for cell gate per layer */
    float **U_c;
    /** Recurrent kernel for output gate per layer */
    float **U_o;
    /** Bias for input gate per layer */
    float **b_i;
    /** Bias for forget gate per layer */
    float **b_f;
    /** Bias for cell gate per layer */
    float **b_c;
    /** Bias for output gate per layer */
    float **b_o;
    /** Layer forward function pointers */
    layer_forward_fcn *layer_forward;
    /** Model finish function pointers */
    model_build_finish_fcn *model_build_finish;
} ModeljoinState;

/** Print a given matrix 
 * @param mat       The matrix
 * @param rows      Number of rows of the matrix
 * @param cols      Number of columns of the matrix
 */
void print_matrix(float *mat, int rows, int cols);

/** Allocate the weight matrices for the model in the state.
 * @param state         The modeljoin state
 * @param num_in_cols   Number of model inputs
 * @param num_layers    Number of model layers
 * @param layer_dims    Layer dimensions
 * @param layertypes    Layer types
 */
void allocate_weights(ModeljoinState *state, int num_in_cols, 
    int num_layers,  int *layer_dims, TFLayerType *layer_types);

/** Free all weight matrices in the modeljoin state.
 * @param state     The modeljoin state
 */
void free_weights(ModeljoinState *state);

/** Copy the inference result to a buffer.
 * @param buf_col       The result buffer column
 * @param result        The inference result
 * @param vectorsize    The vectorsize of the buffer 
 */
void mj_result_to_buffer(float *buf_col, float *result, int vectorsize);

#endif //ML_COMMON_H

