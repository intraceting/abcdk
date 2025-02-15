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

#ifdef AVUTIL_AVUTIL_H

int abcdk_test_cuda(abcdk_option_t *args)
{
    int gpu = abcdk_option_get_int(args,"--gpu",0,0);

    int chk = abcdk_cuda_set_device(gpu);
    assert(chk == 0);
        
    char name[256]= {0};
    chk = abcdk_cuda_get_device_name(name,gpu);
    assert(chk == 0);

    fprintf(stderr,"%s\n",name);

    //AVFrame *a = abcdk_cuda_avframe_alloc(100,1000,AV_PIX_FMT_YUV420P,1);
    //AVFrame *b = abcdk_avframe_alloc(100,1000,AV_PIX_FMT_YUV420P,1);
    //AVFrame *a = abcdk_cuda_avframe_alloc(100,1000,AV_PIX_FMT_NV16,123);
    //AVFrame *b = abcdk_avframe_alloc(100,1000,AV_PIX_FMT_NV16,234);
    //AVFrame *a = abcdk_cuda_avframe_alloc(100,1000,AV_PIX_FMT_NV12,123);
    //AVFrame *b = abcdk_avframe_alloc(100,1000,AV_PIX_FMT_NV12,234);
    AVFrame *a = abcdk_cuda_avframe_alloc(200,200,AV_PIX_FMT_RGB24,123);
    AVFrame *b = abcdk_avframe_alloc(200,200,AV_PIX_FMT_RGB24,234);


    uint8_t scalar[3] = {128,255,0};
    abcdk_cuda_imgproc_stuff_8u_c3r(a->data[0],a->width,a->linesize[0],a->height,scalar);

    uint8_t color[3] = {255,0,0};
    int corner[4] = {10,10,100,100};

    abcdk_cuda_imgproc_drawrect_8u_c3r(a->data[0],a->width,a->linesize[0],a->height,color,3,corner);

    chk = abcdk_cuda_avframe_copy(b,a,1,0);
    assert(chk == 0);

   // abcdk_bmp_save_file("/tmp/test.cuda.a.bmp",a->data[0],a->linesize[0],a->width,a->height,24);
    abcdk_bmp_save_file("/tmp/test.cuda.b.bmp",b->data[0],b->linesize[0],b->width,-b->height,24);

    AVFrame *c = abcdk_avframe_alloc(200,200,AV_PIX_FMT_YUV420P,567);

    abcdk_cuda_avframe_convert(c,a,1,0);

    AVFrame *d = abcdk_cuda_avframe_alloc(200,200,AV_PIX_FMT_YUYV422,567);

    abcdk_cuda_avframe_convert(d,c,0,1);

    av_frame_free(&a);
    av_frame_free(&b);
    av_frame_free(&c);
    av_frame_free(&d);
    
    return 0;
}

#endif //AVUTIL_AVUTIL_H

#else //HAVE_CUDA

int abcdk_test_cuda(abcdk_option_t *args)
{
    return 0;
}

#endif //HAVE_CUDA
