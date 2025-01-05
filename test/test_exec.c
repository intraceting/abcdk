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

// environ

int abcdk_test_exec(abcdk_option_t *args)
{

    int pids[100] = {0};
    int chk = abcdk_file_wholockme("/bin/bash", pids, 100);

    for (int i = 0; i < chk; i++)
    {
        printf("%d\n", pids[i]);
    }

    return 0;
}
