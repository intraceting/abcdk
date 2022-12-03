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
 //   abcdk_log_open("/tmp/test.log","abcdk.%d.log",10,10, 1, 1);
    abcdk_log_open("/tmp/log/aaaa","/tmp/abcdk.%d.log",1,1, 0, 0);

    #pragma omp parallel // for num_threads(1)
    for (int l = ABCDK_LOG_ERROR; l <= ABCDK_LOG_DEBUG; l++)
    {
        for (int i = 0; i < 1000000; i++)
        {
            abcdk_log_printf(l, "test log :%d", i);
          //  sleep(1);
        }    
    }

   //sleep(100);
}