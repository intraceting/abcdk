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
#ifdef HAVE_FFMPEG

int abcdk_test_cuda(abcdk_option_t *args)
{
    int gpu = abcdk_option_get_int(args, "--gpu", 0, 0);

    int chk = abcdk_cuda_set_device(gpu);
    assert(chk == 0);

    char name[256] = {0};
    chk = abcdk_cuda_get_device_name(name, gpu);
    assert(chk == 0);

    fprintf(stderr, "%s\n", name);

    // AVFrame *a = abcdk_cuda_avframe_alloc(100,1000,AV_PIX_FMT_YUV420P,1);
    // AVFrame *b = abcdk_avframe_alloc(100,1000,AV_PIX_FMT_YUV420P,1);
    // AVFrame *a = abcdk_cuda_avframe_alloc(100,1000,AV_PIX_FMT_NV16,123);
    // AVFrame *b = abcdk_avframe_alloc(100,1000,AV_PIX_FMT_NV16,234);
    // AVFrame *a = abcdk_cuda_avframe_alloc(100,1000,AV_PIX_FMT_NV12,123);
    // AVFrame *b = abcdk_avframe_alloc(100,1000,AV_PIX_FMT_NV12,234);
    AVFrame *a = abcdk_cuda_avframe_alloc(200, 200, AV_PIX_FMT_RGB24, 123);
    AVFrame *b = abcdk_avframe_alloc(200, 200, AV_PIX_FMT_RGB24, 234);

    uint8_t scalar[3] = {128, 255, 0};
    abcdk_cuda_imgproc_stuff_8u_c3r(a->data[0], a->width, a->linesize[0], a->height, scalar);

    uint8_t color[3] = {255, 0, 0};
    int corner[4] = {10, 10, 100, 100};

    abcdk_cuda_imgproc_drawrect_8u_c3r(a->data[0], a->width, a->linesize[0], a->height, color, 3, corner);

    chk = abcdk_cuda_avframe_copy(b, a);
    assert(chk == 0);

    // abcdk_bmp_save_file("/tmp/test.cuda.a.bmp",a->data[0],a->linesize[0],a->width,a->height,24);
    abcdk_cuda_avframe_save("/tmp/test.cuda.b.bmp", b);

    AVFrame *c = abcdk_cuda_avframe_alloc(200, 200, AV_PIX_FMT_YUV420P, 567);

    abcdk_cuda_avframe_convert(c, a);

    AVFrame *d = abcdk_cuda_avframe_alloc(200, 200, AV_PIX_FMT_RGB24, 678);

    abcdk_cuda_avframe_convert(d, c);

    AVFrame *e = abcdk_cuda_avframe_alloc(800, 600, AV_PIX_FMT_RGB24, 678);

    abcdk_cuda_avframe_resize(e, NULL, d, NULL, 1, NPPI_INTER_CUBIC);

    abcdk_cuda_avframe_save("/tmp/test.cuda.e.bmp", e);

    AVFrame *f = abcdk_cuda_avframe_alloc(800, 600, AV_PIX_FMT_RGB24, 678);

    NppiPoint dst_quad[4] = {
        {30, 30},   // 变换后的左上角
        {220, 50},  // 变换后的右上角
        {210, 220}, // 变换后的右下角
        {50, 230},  // 变换后的左下角
    };

    NppiRect src_roi = {100, 100, 200, 200};

    // abcdk_cuda_avframe_warp(f, NULL, dst_quad , e, &src_roi , NULL,2 , NPPI_INTER_CUBIC);
    abcdk_cuda_avframe_warp(f, NULL, dst_quad, e, NULL, NULL, 1, NPPI_INTER_CUBIC);

    abcdk_cuda_avframe_save("/tmp/test.cuda.f.bmp", f);

    abcdk_cuda_jpeg_save("/tmp/test.cuda.f.jpg", f);

    for (int i = 0; i < 100; i++)
    {
        AVFrame *g = abcdk_cuda_jpeg_load("/tmp/test.cuda.f.jpg");

        abcdk_cuda_jpeg_save("/tmp/test.cuda.g.jpg", g);

        av_frame_free(&g);
    }

    av_frame_free(&a);
    av_frame_free(&b);
    av_frame_free(&c);
    av_frame_free(&d);
    av_frame_free(&e);
    av_frame_free(&f);
    

    return 0;
}

#endif // HAVE_FFMPEG
#else  // HAVE_CUDA

int abcdk_test_cuda(abcdk_option_t *args)
{
    return 0;
}

#endif // HAVE_CUDA
