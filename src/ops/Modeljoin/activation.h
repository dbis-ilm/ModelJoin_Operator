#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "mj_common.h"


/** Apply activation function on input vector.
 * @param vec           Data vector
 * @param n             Size of data vector
 * @param activation    Activation funciton identifier
 *//** Apply activation function on input vector.
 * @param vec           Data vector
 * @param n             Size of data vector
 * @param activation    Activation funciton identifier
 */
void vector_activate(float *vec, int n, TFActivationFunction activation);

#ifdef USECUDA
/** Apply activation function on input vector using cuda kernel.
 * @param vec           Data vector
 * @param n             Size of data vector
 * @param activation    Activation funciton identifier
 */
void vector_activate_cuda(float *vec, int n, TFActivationFunction activation);
#endif

#endif //ACTIVATION_H