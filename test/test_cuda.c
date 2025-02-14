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

// environ

#ifdef HAVE_CUDA

int abcdk_test_cuda(abcdk_option_t *args)
{
    int gpu = abcdk_option_get_int(args,"--gpu",0,0);

    int chk = abcdk_cuda_set_device(gpu);
    assert(chk == 0);
        
    char name[256]= {0};
    chk = abcdk_cuda_get_device_name(name,gpu);
    assert(chk == 0);

    fprintf(stderr,"%s\n",name);

    void *gptr = abcdk_cuda_alloc(11);


    abcdk_cuda_free(&gptr);
    

    
    return 0;
}

#else //HAVE_CUDA

int abcdk_test_cuda(abcdk_option_t *args)
{
    return 0;
}

#endif //HAVE_CUDA
