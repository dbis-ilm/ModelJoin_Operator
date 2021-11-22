#include "src/engine.h"
#include "test/paths.h"
int main() {
    init_paths();
    engine_start();

    Type sinus_schema[] = {FLOAT, FLOAT, FLOAT};
    Operator *data = scan_build(sinus, "|", 3, sinus_schema, test_vectorsize);

    Type model_schema[] = {INT, INT, INT, INT, FLOAT, FLOAT, FLOAT, FLOAT, 
        FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT};
    Operator *model = scan_build(sinus_model, "|", 16, model_schema, test_vectorsize);

    int in_cols[] = {0, 1, 2};
    int layer_dims[] = {32, 50, 1};
    TFLayerType layer_types[] = {LSTM, DENSE, DENSE};
    TFActivationFunction layer_activations[] = {LINEAR, RELU, LINEAR};
    Operator *join = modeljoin_build(model, data, 3, in_cols, 3, layer_dims, layer_types, layer_activations);

    run_query(join, true);
    print_query_profile(join);
    query_close(join);

    engine_stop();
}
