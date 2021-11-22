#include <math.h>
#include "activation.h"

inline static void relu_activation(float *vec, int n)
{
    int i; 
    for (i = 0; i < n; i++) vec[i] *= vec[i] > 0;
}

inline static void sigmoid_activation(float *vec, int n)
{
    int i; 
    for (i = 0; i < n; i++) vec[i] = 1/(1+exp(-vec[i]));
}

inline static void tanh_activation(float *vec, int n)
{
    int i; 
    for (i = 0; i < n; i++) vec[i] = tanh(vec[i]);
}

void vector_activate(float *vec, int n, TFActivationFunction activation)
{
    switch (activation){
        case LINEAR: break;
        case RELU: 
            relu_activation(vec, n);
            break;
        case SIGMOID: 
            sigmoid_activation(vec, n);
            break;
        case TANH:
            tanh_activation(vec, n);
            break;
    }
}