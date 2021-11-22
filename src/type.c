#include "type.h"

int sizeof_Type(Type t) {
    int ret;
    switch (t)
    {
    case INT:
        ret = sizeof(long);
        break;
    case FLOAT:
        ret = sizeof(float);
        break;
    case STRING:
        ret = sizeof(char*);
        break;
    default:
        assert(0);
        break;
    }
    return ret;
}