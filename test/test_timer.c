/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

static void _routine_cb(void *opaque)
{
    printf("ms:%llu\n",abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 3));
    sleep(1);
}


int abcdk_test_timer(abcdk_option_t *args)
{
    abcdk_timer_t *ctx = abcdk_timer_create(5*1000,_routine_cb,NULL);

    getchar();

    abcdk_timer_destroy(&ctx);

    return 0;
}