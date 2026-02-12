/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

static int oscillating_random(int *numbers, int size) {
    double offset = sin(abcdk_time_clock2kind_with(CLOCK_MONOTONIC,9) * 0.1) * (size / 2);
    int index = (int)(size / 2 + offset);
    return numbers[index % size];
}

static int test_oscillating_random()
{
    srand(time(NULL));

    int numbers[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int size = sizeof(numbers) / sizeof(numbers[0]);

    for(int i = 0;i<10;i++)
    {
        int random_number = oscillating_random(numbers, size);
        printf("Generated number: %d\n", random_number);
    }

    return 0;
}


int abcdk_test_rand(abcdk_option_t *args)
{
    test_oscillating_random();
}