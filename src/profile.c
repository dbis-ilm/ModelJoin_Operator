#include "profile.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

const int kProfileObjectCount = EXECUTE + 1;

static bool timeval_is_uninitialised(struct timeval t) {
    return t.tv_sec == -1;
}

static void timeval_uninitialise(struct timeval *t) {
    t->tv_sec = -1;
}

ProfileStats *profile_alloc() {
    ProfileStats *ps = malloc(sizeof(ProfileStats));
    ps->stats = calloc(kProfileObjectCount, sizeof(double));
    timeval_uninitialise(&(ps->current));
    return ps;
}

void profile_start(ProfileStats *ps) {
    if (!timeval_is_uninitialised(ps->current)) {
        fprintf(stderr, "Called profile_start although last measurement isn't finished. \n");
        assert(0);
    }
    gettimeofday(&(ps->current), NULL);
}

void profile_stop(ProfileStats *ps, ProfileObject obj) {
    struct timeval now, diff;
    if (timeval_is_uninitialised(ps->current)) {
        fprintf(stderr, "Called profile_end although measurement isn't started. \n");
        assert(0);
    }
    gettimeofday(&now, NULL);
    timersub(&now, &(ps->current), &diff);
    ps->stats[obj] += (double) diff.tv_usec / 1000000 + diff.tv_sec;
    timeval_uninitialise(&(ps->current));
}

static char* profile_get_object_name(ProfileObject obj) {
    switch(obj) {
        case BUILD: return "BUILD";
        case EXECUTE: return "EXECUTE";
        default: assert(0);
    }
}

void profile_print(ProfileStats *ps, const char* operator_name, ProfileStats *total) {
    int i;
    printf("%s | ", operator_name);
    for(i = 0; i < kProfileObjectCount - 1; i++) {
        printf("%s: %f s, ", profile_get_object_name(i), ps->stats[i]);
        if (total) total->stats[i] += ps->stats[i];
    }
    printf("%s: %f s\n", profile_get_object_name(i), ps->stats[i]);
    if (total) total->stats[i] += ps->stats[i];
}

void profile_free(ProfileStats *ps) {
    free(ps->stats);
    free(ps);
}

