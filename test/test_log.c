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

int abcdk_test_log(abcdk_option_t *args)
{
   abcdk_logger_t *ctx = abcdk_logger_open("/tmp/abcdk/log/test.log","test.%d.log",10,10, 1, 1);
 //   abcdk_logger_t *ctx = abcdk_logger_open("/tmp/abcdk/log/test","/tmp/abcdk/log/test.%d.log",1,1, 0, 0);

    #pragma omp parallel // for num_threads(1)
    for (int l = ABCDK_LOGGER_ERROR; l <= ABCDK_LOGGER_DEBUG; l++)
    {
        for (int i = 0; i < 1000000; i++)
        {
            abcdk_logger_printf(ctx,l, "test log :%d\naaaa%d\nbbbbbb\ncccc", i,i-1000000);
          //  sleep(1);
        }    
    }

   //sleep(100);

   abcdk_logger_close(&ctx);

   return 0;
}
