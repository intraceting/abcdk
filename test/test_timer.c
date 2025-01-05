/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

static uint64_t _routine_cb(void *opaque)
{
    printf("ms=%.06f,len=%llu\n",(double)abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 6)/1000000.,abcdk_rand(50,1234));
    //return abcdk_rand(1000,3000);
    return abcdk_rand(1,3);
}


int abcdk_test_timer(abcdk_option_t *args)
{
    abcdk_timer_t *ctx = abcdk_timer_create(_routine_cb,NULL);

    getchar();

    abcdk_timer_destroy(&ctx);

    return 0;
}