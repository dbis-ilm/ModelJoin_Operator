#ifndef PROFILE_H
#define PROFILE_H

#include <sys/time.h>

/** Struct holding profiling information */
struct ProfileStats_t {
    /** Array of timings for ProfileObjects */
    double *stats;
    /** Start timer, initialized if timing is running, uninitialized otherwise */
    struct timeval current;
};

/** Step of query execution that can be profiled */
typedef enum ProfileObject_t { 
    BUILD = 0,
    EXECUTE = 1
} ProfileObject;

typedef struct ProfileStats_t ProfileStats;

/** Allocate a ProfileStats object, setting all timers to zero.
 * @return allocated ProfileStats
 */
ProfileStats *profile_alloc();

/** Start the timing.
 * @param ps    ProfileStats object to store timing in 
 */
void profile_start(ProfileStats *ps);

/** Stop the timing and add difference to the given execution step.
 * @param ps    ProfileStats object to store timing in.
 * @param obj   Execution step
 */
void profile_stop(ProfileStats *ps, ProfileObject obj);

/** Print the profile of an operator.
 * @param ps                ProfileStats object of an operator.
 * @param operator_name     Name to be printed
 * @param total             Used to sum up the steps for the whole query, if not NULL.
 */
void profile_print(ProfileStats *ps, const char* operator_name, ProfileStats *total);

/** Free the profile stats.
 * @param ps                ProfileStats to be freed.
 */
void profile_free(ProfileStats *ps);

#endif //PROFILE_H