#include "../src/engine.h"
#include "paths.h"
int main() {
    init_paths();
    engine_start();

    int vectorsize = test_vectorsize;

    Type sinus_schema[] = {FLOAT, FLOAT, FLOAT};
    Operator *data = scan_build(sinus, "|", 3, sinus_schema, vectorsize);

    Type model_schema[] = {INT, INT, INT, INT, FLOAT, FLOAT, FLOAT, FLOAT, 
        FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT};
    Operator *model = scan_build(sinus_model, "|", 16, model_schema, vectorsize);

    int in_cols[] = {0, 1, 2};
    int layer_dims[] = {32, 50, 1};
    TFLayerType layer_types[] = {LSTM, DENSE, DENSE};
    TFActivationFunction layer_activations[] = {LINEAR, RELU, LINEAR};
    Operator *join = modeljoin_build(model, data, 3, in_cols, 3, layer_dims, layer_types, layer_activations);

    Type expected_schema[] = {FLOAT, FLOAT, FLOAT, FLOAT};
    Operator *expected = scan_build(sinus_expected, "|", 4, expected_schema, vectorsize);
    Operator *compare = compare_build(join, expected);

    run_query(compare, true);
    print_query_profile(compare);
    query_close(compare);

    engine_stop();
}