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

int abcdk_test_log(abcdk_tree_t *args)
{
    abcdk_log_open(NULL, 1, 1);

    #pragma omp parallel for num_threads(3)
    for (int l = ABCDK_LOG_ERROR; l <= ABCDK_LOG_DEBUG; l++)
    {
        for (int i = 0; i < 100000; i++)
        {
            abcdk_log_printf(l, "test log :%d", i);
          //  sleep(1);
        }    
    }

    sleep(100);
}