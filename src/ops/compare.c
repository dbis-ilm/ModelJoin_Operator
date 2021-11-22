#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include "../operator.h"

typedef int (*vector_compare_fcn)(char **data1, char **data2, int vectorsize);

typedef struct CompareState_t {
    int failed;
    vector_compare_fcn *cmp_fcns;
} CompareState;

static void compare_close(Operator *op)
{
    CompareState *state = (CompareState*) op->state;
    int failed = state->failed;
    free(state->cmp_fcns);
    free(state);
    operator_free(op);
    if (failed != 0) {
        printf("%i Column values did not match\n", failed);
        assert(false);
    } else {
        printf("Results matched\n");
    }
}

int compare_int(char **data1, char **data2, int vectorsize)
{
    int *d1 = (int*) data1;
    int *d2 = (int*) data2;
    int i;
    int failed = 0;
    for (i = 0; i < vectorsize; i++) {
        if (d1[i] != d2[i]) failed++;
    }
    return failed;
}

int compare_float(char **data1, char**data2, int vectorsize)
{
    float *d1 = (float*) data1;
    float *d2 = (float*) data2;
    int i;
    int failed = 0;
    for (i = 0; i < vectorsize; i++) {
        if (fabs(d1[i] - d2[i]) > 0.000001) failed++;
    }
    return failed;
}

int compare_string(char **data1, char**data2, int vectorsize)
{
    int i;
    int failed = 0;
    for (i = 0; i < vectorsize; i++) {
        if (strcmp(data1[i], data2[i]) != 0) failed++;
    }
    return failed;
}

static int compare_buffers(Buffer b1, Buffer b2, vector_compare_fcn *cmp_fcns, int num_cols, int vectorsize)
{
    int i;
    int failed = 0;
    for (i = 0; i < num_cols; i++) {
        failed += cmp_fcns[i](b1[i], b2[i], vectorsize);
    }
    
    return failed;
}

static int compare_next(Operator *op)
{
    CompareState *state = (CompareState*) op->state;
    int n1, n2;
    
    n1 = op->lchild->next(op->lchild);
    n2 = op->rchild->next(op->rchild);

    while (n1 != 0 || n2 != 0) {
        Buffer d1 = op->lchild->data;
        Buffer d2 = op->rchild->data;
        int failed;

        assert((n1 == n2) && "Number of rows do not match");

        failed = compare_buffers(d1, d2, state->cmp_fcns, op->lchild->num_cols, n1);

        if (failed != 0) {
            printf("Errors in buffers: \n\n");
            print_buffer(d1, op->lchild->col_types, op->lchild->num_cols, n1);
            printf("\n");
            print_buffer(d2, op->rchild->col_types, op->rchild->num_cols, n2);
            printf("\n--------------------------------------------\n\n");
            state->failed += failed;
        }

        n1 = op->lchild->next(op->lchild);
        n2 = op->rchild->next(op->rchild); 
    }

    /* Allocate an empty buffer so that run_query succeeds */
    op->data = allocate_buffer(op->num_cols, op->col_types, op->lchild->vectorsize);

    return 0;
}

Operator *compare_build(Operator *left, Operator *right)
{
    int i;
    int num_cols = left->num_cols;
    Operator *op;
    Type ret_types[] = {STRING};
    CompareState *state;
    assert((left->num_cols == right->num_cols) && "Number of columns do not match");
    assert((left->vectorsize == right->vectorsize) && "Vectorsizes do not match");

    for (i = 0; i < num_cols; i++) assert((left->col_types[i] == right->col_types[i]) && "Column types do not match");

    op = operator_alloc(&compare_next, &compare_close, left, right, 1, ret_types, "Compare");
    op->vectorsize = left->vectorsize;
    op->state = malloc(sizeof(CompareState));
    state = (CompareState*) op->state;
    state->failed = 0;
    state->cmp_fcns = malloc(num_cols * sizeof(vector_compare_fcn));
    for (i = 0; i < num_cols; i++) {
        switch(left->col_types[i]) {
            case INT:
                state->cmp_fcns[i] = &compare_int;
                break;
            case FLOAT:
                state->cmp_fcns[i] = &compare_float;
                break;
            case STRING:
                state->cmp_fcns[i] = &compare_string;
                break;
        }
    }
    return op;
}