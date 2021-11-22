#ifndef TYPE_H
#define TYPE_H

#include<assert.h>

/** Basic enumeration of supported types */
typedef enum Type_t { 
    INT = 1,
    FLOAT = 2,
    STRING = 3
} Type;

#define kmaxTypeValue STRING

/** Size of the type in bytes
 * @param t     Type
 * @return size of type in bytes
 */
int sizeof_Type(Type t);

#endif 