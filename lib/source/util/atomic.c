/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/atomic.h"

static pthread_mutex_t g_atomic_lock = PTHREAD_MUTEX_INITIALIZER;

void abcdk_atomic_lock(void)
{
    pthread_mutex_lock(&g_atomic_lock);
}

void abcdk_atomic_unlock(void)
{
    pthread_mutex_unlock(&g_atomic_lock);
}