#include "engine.h"
#include "utils/stack.h"

#ifdef PROFILE
#include "profile.h"
#endif

void run_query(Operator *root, bool print) {
    int n = root->next(root);
    long num_results = n;
    while (n > 0) {
        if (print) print_buffer(root->data, root->col_types, root->num_cols, n);
        free_buffer(root->data, root->num_cols, root->col_types, n);
        n = root->next(root);
        num_results += n;
    }
    free_buffer(root->data, root->num_cols, root->col_types, n); // Free the last empty buffer
    printf("(%li results)\n", num_results);
}

void print_query_profile(Operator *root) {
#ifdef PROFILE
    Operator *op = root;
    Stack *stack = stack_create();
    ProfileStats *total = profile_alloc();
    assert(root);

    printf("\nQuery Profile (Vectorsize: %i) \n", root->vectorsize);
    printf("************* \n");

    while (op) {
        /* TODO: indentation for subtrees? */
        profile_print(op->profile_stats, op->name, total);
        if (op->rchild) stack_push(stack, (char*)op->rchild);
        op = op->lchild;
        if (!op) op = (Operator*) stack_pop(stack);
    }
    printf("------------- \n");
    profile_print(total, "TOTAL", NULL);
    profile_free(total);
#else
    printf("No profile information collected\n");
#endif
}

void query_close(Operator *root) {
    Operator *op = root;
    Stack *stack = stack_create();
    while (op) {
        Operator *child = op->lchild;
        if (op->rchild) stack_push(stack, (char*) op->rchild);
        op->close(op);
        op = child;
        if (!op) op = (Operator*) stack_pop(stack);
    }
}

void engine_start() {
#ifdef USECUDA
    cublas_init();
#endif
    return;
}

void engine_stop() {
#ifdef USECUDA
    cublas_exit();
#endif
    return;
}