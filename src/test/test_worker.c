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

void abcdk_test_worker_routine(void *opaque,uint64_t event,void *item)
{
    int *id = (int *)opaque;
    int a = 123, b = 345, c = 456;

    fprintf(stderr,"event-begin: %d\n",event);

    for (int h = 0; h < 2160; h++)
    {
        for (int w = 0; w < 3840; w++)
        {

            if (w < 3000)
            {
                a += a;
                b += b;
                c += c;
            }
            else 
            {
                a *= a;
                b *= b;
                c *= c;
            }
            
        }
    }

    fprintf(stderr,"event-end: %d\n",event);
}

int abcdk_test_worker(abcdk_option_t *args)
{
    int threads = abcdk_option_get_int(args,"--threads",0,4);

    abcdk_worker_config_t cfg = {threads,0,0,0,0,NULL,abcdk_test_worker_routine};
    abcdk_worker_t *ctx = abcdk_worker_start(&cfg);

    for(int i = 0;i<10;i++)
    {
        abcdk_worker_dispatch(ctx,i,NULL);
    }

    abcdk_worker_stop(&ctx);

    return 0;

}
