#ifndef DENSE_H
#define DENSE_H

#ifdef USECUDA
/** Consume a dense layer tuple from the model table, being part of the modeljoin build phase. CUDA implementation.
 * @param state             ModelJoin state holding pointers to model weights
 * @param logical_layer     Logical layer offset
 * @param data              Buffer from the model table scan
 * @param row               row offset of the tuple in the buffer
 */
void dense_layer_consume_tuple_cuda(ModeljoinState *state, int logical_layer, Buffer data, int row);

/** Function to call on a dense layer after building is finished. CUDA implementation.
 * @param op        Pointer to the modeljoin operatpr
 * @param layer     Logical layer of/** Consume a dense layer tuple from the model table, being part of the modeljoin build phase. CUDA implementation.
 * @param state             ModelJoin state holding pointers to model weights
 * @param logical_layer     Logical layer offset
 * @param data              Buffer from the model table scan
 * @param row               row offset of the tuple in the buffer
 */fset
 */
void dense_layer_finish_cuda(void *op, int layer);

/** Dense layer forward function using matrix memcopy loading, performing matrix operations on the intermediate result if available.
 * CUDA implementation.
 * @param op                Pointer to the modeljoin operator
 * @param layer             Logical layer offset
 * @param intermediate      Intermediate result from preceeding layer if exists.
 * @param int_rows          Number of rows of intermediate result.
 * @param int_cols          Number of cols of intermediate result.
 * @return                  new intermediate result and sets int_rows and int_cols
 */
float* dense_layer_forward_matrix_memcpy_loading_cuda(void *op, int layer, float *intermediate, int *int_rows, int *int_cols);

#else
/** Consume a dense layer tuple from the model table, being part of the modeljoin build phase.
 * @param state             ModelJoin state holding pointers to model weights
 * @param logical_layer     Logical layer offset
 * @param data              Buffer from the model table scan
 * @param row               row offset of the tuple in the buffer
 */
void dense_layer_consume_tuple(ModeljoinState *state, int logical_layer, Buffer data, int row);

/** Function to call on a dense layer after building is finished.
 * @param op        Pointer to the modeljoin operatpr
 * @param layer     Logical layer offset
 */
void dense_layer_finish(void *op, int layer);

/** Dense layer forward function using rowwise loading, performing matrix operations on the intermediate result if available.
 * @param op                Pointer to the modeljoin operator
 * @param layer             Logical layer offset
 * @param intermediate      Intermediate result from preceeding layer if exists.
 * @param int_rows          Number of rows of intermediate result.
 * @param int_cols          Number of cols of intermediate result.
 * @return                  new intermediate result and sets int_rows and int_cols
 */
float* dense_layer_forward_rowwise(void *op, int layer, float *intermediate, int *int_rows, int *int_cols);

/** Dense layer forward function using manual matrix loading, performing matrix operations on the intermediate result if available.
 * @param op                Pointer to the modeljoin operator
 * @param layer             Logical layer offset
 * @param intermediate      Intermediate result from preceeding layer if exists.
 * @param int_rows          Number of rows of intermediate result.
 * @param int_cols          Number of cols of intermediate result.
 * @return                  new intermediate result and sets int_rows and int_cols
 */
float* dense_layer_forward_matrix_manual_loading(void *op, int layer, float *intermediate, int *int_rows, int *int_cols);

/** Dense layer forward function using memcpy loading, performing matrix operations on the intermediate result if available.
 * @param op                Pointer to the modeljoin operator
 * @param layer             Logical layer offset
 * @param intermediate      Intermediate result from preceeding layer if exists.
 * @param int_rows          Number of rows of intermediate result.
 * @param int_cols          Number of cols of intermediate result.
 * @return                  new intermediate result and sets int_rows and int_cols
 */
float* dense_layer_forward_matrix_memcpy_loading(void *op, int layer, float *intermediate, int *int_rows, int *int_cols);
#endif // USECUDA

#endif // DENSE_H