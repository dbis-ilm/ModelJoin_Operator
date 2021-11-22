#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include "stack.h"

struct Stack {
    int top;
    int allocated;
    int filled;
    char** data;
};

Stack *stack_create()
{
    Stack *stack = (Stack*)malloc(sizeof(Stack));
    stack->top = -1;
    stack->allocated = 4;
    stack->filled = 0;
    stack->data = (char**)malloc(stack->allocated * sizeof(char*));
    return stack;
}

static bool isFull(Stack *stack)
{
    return stack->filled == stack->allocated;
}
 
static bool isEmpty(Stack *stack)
{
    return stack->top == -1;
}
 
static void double_stack(Stack *stack) {
    char **new_data = realloc(stack->data, stack->allocated * 2 * sizeof(char*));
    assert(new_data);
    stack->data = new_data;
    stack->allocated *= 2;
}

void stack_push(Stack *stack, char *item)
{
    if (isFull(stack))
        double_stack(stack);
    stack->data[++stack->top] = item;
}
 
char* stack_pop(struct Stack* stack)
{
    if (isEmpty(stack))
        return NULL;
    return stack->data[stack->top--];
}