#include <stdlib.h>
#include <string.h>

// VECTORSIZE
int test_vectorsize = 8;
int exp_vectorsize = 1024;

// DATA
char iris[1024];
char iris_expected[1024];
char iris_model[1024];
char sinus[1024];
char sinus_expected[1024];
char sinus_model[1024];

void init_paths() {
    char *test_dir = getenv("TEST_DIR");
    assert(test_dir);
   
    strcpy(iris, test_dir);
    strcat(iris, "/data/iris.csv");

    strcpy(iris_expected, test_dir);
    strcat(iris_expected, "/data/iris.expected");

    strcpy(iris_model, test_dir);
    strcat(iris_model, "/data/iris_model.csv");

    strcpy(sinus, test_dir);
    strcat(sinus, "/data/sinus.csv");

    strcpy(sinus_expected, test_dir);
    strcat(sinus_expected, "/data/sinus.expected");

    strcpy(sinus_model, test_dir);
    strcat(sinus_model, "/data/sinus_model.csv");

}
