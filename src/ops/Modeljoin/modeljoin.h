#ifndef MODELJOIN_H
#define MODELJOIN_H

#include "../../operator.h"
#include "mj_common.h"

/** Build a model join operator.
 * @param model         Pointer to the scan of the model table
 * @param child         Query subtree to apply the model on
 * @param num_in_cols   Number of model input columns
 * @param arg_col_map   Input column offsets
 * @param num_layers    Number of layers in the model table
 * @param layer_dims    Number of nodes per layer
 * @param layer_types   Types of the model layers
 * @return Pointer to the operator
 */
Operator *modeljoin_build(Operator *model, Operator *child, int num_in_cols, int *arg_col_map, 
    int num_layers, int *layer_dims, TFLayerType *layer_types, TFActivationFunction *layer_activations);

#endif //MODELJOIN_H