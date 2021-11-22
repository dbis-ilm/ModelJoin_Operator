#ifndef BUFFER_H
#define BUFFER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h> 
#include "type.h"

/** Basic buffer layout is a array of columns, with a column being an array of pointers or values. 
 * String are stored out-of-place, referenced by a pointer from the buffer. Other types are stored
 * in-place and require pointer casting.
*/
typedef char*** Buffer;

/** Allocate a new buffer.
 * @param num_cols      Number of columns
 * @param types         Column types, indicating the value width.
 * @param vectorsize    Number of elements per column
 * @return allocated buffer
 */
Buffer allocate_buffer(int num_cols, Type* types, int vectorsize);

/** Free a buffer, including out-of-place stored strings if present.
 * @param buf           Buffer to be freed
 * @param num_cols      Number of columns
 * @param types         Column types, indicating the value width.
 * @param n             Number of elements per column
 */
void free_buffer(Buffer buf, int num_cols, Type *types, int n);

/** Add or remove columns from a buffer.
 * @param buf           Buffer to reallocate
 * @param old_cols      Old number of columns
 * @param new_cols      New number of columns
 * @param types         Column types, indicating the value width.
 * @param vectorsize    Number of elements per column
 * @return reallocated buffer
 */
Buffer reallocate_buffer(Buffer buf, int old_cols, int new_cols, Type *types, int vectorsize);

/** Print a buffer.
 * @param buf           Buffer to print
 * @param types         Column types, indicating the value width.
 * @param num_cols      Number of columns
 * @param vectorsize    Number of elements per column
 */
void print_buffer(Buffer buf, Type *types, int num_cols, int vectorsize);

#endif 