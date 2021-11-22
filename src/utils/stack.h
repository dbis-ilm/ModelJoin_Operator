typedef struct Stack Stack;

Stack *stack_create();

void stack_push(Stack *stack, char *item);

char* stack_pop(struct Stack* stack);