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

    assert(seed != NULL && *seed != 0);

    next = *seed;

    /*POSIX.1-2001*/

    next = next * 1103515245UL + 12345UL;
    next = next / 65536;

    /*copy.*/
    *seed = next;

    return next;
}