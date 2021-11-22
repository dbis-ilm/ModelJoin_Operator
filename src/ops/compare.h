#ifndef COMPARE_H
#define COMPARE_H

#include "../operator.h"

/** Build a compare operator validating that data streams are equal, and otherwise asserts. Used for testing.
 * @param left      left child operator
 * @param right     right child operator
 * @return Pointer to the operator
 */
Operator *compare_build(Operator *left, Operator *right);

#endif //COMPARE_H