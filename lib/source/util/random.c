/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/util/random.h"

int64_t abcdk_rand(uint64_t *seed)
{
    uint64_t next;

    assert(seed != NULL);

    if(*seed == 0)
        *seed = abcdk_time_clock2kind_with(CLOCK_MONOTONIC,6);

    next = *seed;

    /*POSIX.1-2001*/

    next = next * 1103515245UL + 12345UL;
    next = next / 65536;

    /*copy.*/
    *seed = next;

    return next;
}

int64_t abcdk_rand_q()
{
    static uint64_t seed = 0;
    int64_t num;

    abcdk_atomic_lock();
    num = abcdk_rand(&seed);
    abcdk_atomic_unlock();

    return num;
}