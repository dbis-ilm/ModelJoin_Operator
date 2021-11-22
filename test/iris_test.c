#include "../src/engine.h"
#include "paths.h"
int main() {
    init_paths();
    engine_start();

    int vectorsize = test_vectorsize;

    Type iris_schema[] = {FLOAT, FLOAT, FLOAT, FLOAT, STRING};
    Operator *data = scan_build(iris, "|", 5, iris_schema, vectorsize);

    Type model_schema[] = {INT, INT, INT, INT, FLOAT, FLOAT, FLOAT, FLOAT, 
        FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT};
    Operator *model = scan_build(iris_model, "|", 16, model_schema, vectorsize);

    int in_cols[] = {0, 1, 2, 3};
    int layer_dims[] = {64, 8, 2, 1};
    TFLayerType layer_types[] = {DENSE, DENSE, DENSE, DENSE};
    TFActivationFunction layer_activations[] = {LINEAR, RELU, SIGMOID, LINEAR};
    Operator *join = modeljoin_build(model, data, 4, in_cols, 4, layer_dims, layer_types, layer_activations);

    Type expected_schema[] = {FLOAT, FLOAT, FLOAT, FLOAT, STRING, FLOAT};
    Operator *expected = scan_build(iris_expected, "|", 6, expected_schema, vectorsize);
    Operator *compare = compare_build(join, expected);

    run_query(compare, true);
    print_query_profile(compare);
    query_close(compare);

    engine_stop();
}