#ifndef ENGINE_H
#define ENGINE_H

#include <stdbool.h>
#include "operator.h"


/** Run a query by calling the next function repeatedly on the root operator. 
 * @param root  The root operator of the query.
 * @param print Indicator if the result should be printed.
 */
void run_query(Operator *root, bool print);

/** Print the profile of the whole query by following the operator tree and 
 * printing the profile of each operator.
 * @param root  The root operator of the query.
 */
void print_query_profile(Operator *root);

/** Free all operators of the query tree.
 * @param root  The root operator of the query.
 */
void query_close(Operator *root);

/** Initialize the engine. Must be called before first query. */
void engine_start();

/** Stop the engine.*/
void engine_stop();

#endif //ENGINE_H