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

void abcdk_test_parallel_routine(void *opaque, uint32_t tid)
{
    int *id = (int *)opaque;
    int a = 123, b = 345, c = 456;

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
}

int abcdk_test_parallel(abcdk_option_t *args)
{

    int threads = abcdk_option_get_int(args,"--threads",0,4);
    abcdk_parallel_t *ctx = abcdk_parallel_alloc(threads);

    for (int i = 0; i < 100; i++)
    {
        uint64_t s = 0;

        abcdk_clock(s, &s);

        abcdk_parallel_run(ctx, i+1, NULL, abcdk_test_parallel_routine);

        u_int64_t s2 = abcdk_clock(s, &s);
        printf("%d = %llu\n", threads, s2);
    }

    abcdk_parallel_free(&ctx);

}