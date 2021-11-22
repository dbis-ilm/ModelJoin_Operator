#include "scan.h"

/** State of the scan operator */
struct ScanState_t {
    /** File handle*/
    FILE *f;
    /** Input file path */
    char *filepath;
    /** Delimiter symbol within file */
    char *delimiter;
};

typedef struct ScanState_t ScanState;

void scan_close(Operator *op) {
    ScanState *state = (ScanState*) op->state;
    if (state->f) {
        fclose(state->f);
    }
    operator_free(op);
}

static void cast_and_fill(char **column, int position, char *value, Type t) {
    
    switch (t)
    {
    case INT:
        ((long*)column)[position] = strtol(value, NULL, 10);
        break;

    case FLOAT:
        ((float*)column)[position] = strtof(value, NULL);
        break;

    case STRING:
        column[position] = calloc(strlen(value) + 1, sizeof(char));
        strncpy(column[position], value, strlen(value));
        break;

    default:
        assert(0);
        break;
    }
}

static int scan_next(Operator *op) {
    ScanState *state = (ScanState*) op->state;
    int max_line_size = op->num_cols * 1024;
    char *line;
    int lines = 0;
    int i;
    char *val;
#ifdef PROFILE
    profile_start(op->profile_stats);
#endif
    op->data = allocate_buffer(op->num_cols, op->col_types, op->vectorsize);
    line = malloc(max_line_size * sizeof(char));
    
    while (lines < op->vectorsize && fgets(line, max_line_size, state->f)) {
        line[strcspn(line, "\n")] = 0; // remove newline character
        val = strtok(line, state->delimiter);
        if (val) cast_and_fill(op->data[0], lines, val, op->col_types[0]);

        for (i = 1; i < op->num_cols; i++) {
            val = strtok(NULL, state->delimiter);
            if (val) cast_and_fill(op->data[i], lines, val, op->col_types[i]);
        }
        lines++;
    }
    free(line);
#ifdef PROFILE
    profile_stop(op->profile_stats, EXECUTE);
#endif
    return lines;
}

Operator *scan_build(char *filepath, char *delimiter, int num_cols, Type *types, int vectorsize) {
    Operator *op = operator_alloc(&scan_next, &scan_close, NULL, NULL, num_cols, types, "FileScan");
    ScanState *state;
    op->vectorsize = vectorsize;
    op->state = malloc(sizeof(ScanState));
    state = ((ScanState*)(op->state));
    state->filepath = filepath;
    state->delimiter = delimiter;
    state->f = fopen(state->filepath, "r");

    assert(state->f);
#ifdef PROFILE
    profile_stop(op->profile_stats, BUILD);
#endif
    return op;
}