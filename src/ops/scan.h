#ifndef SCAN_H
#define SCAN_H

#include "../operator.h"

/** Build a scan operator, which is a leaf node of the query tree. Allocates and fills columnwise structured
 * buffers of vectorsize elements.
 * @param filepath      Structured input file
 * @param delimiter     Delimiter symbol in input file
 * @param num_cols      Expected number of columns in file
 * @param types         Column types
 * @param vectorsize    Number of tuples per output buffer
 * @return Pointer to the operator
 */
Operator *scan_build(char *filepath, char *delimiter, int num_cols, Type *types, int vectorsize);

#endif //SCAN_H