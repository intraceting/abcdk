/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

int abcdk_test_stitcher_cpu(abcdk_option_t *args)
{

    abcdk_opencv_stitcher_t *ctx = abcdk_opencv_stitcher_create(ABCDK_TORCH_TAG_HOST);

    abcdk_object_t *metadata = abcdk_object_copyfrom_file(abcdk_option_get(args, "--metadata-load", 0, ""));

    if (metadata)
    {
        abcdk_opencv_stitcher_metadata_load(ctx, "7b1a5e0796419a8278e1f6df640f0bfb", metadata->pstrs[0]);
        abcdk_object_unref(&metadata);
    }

    metadata = abcdk_opencv_stitcher_metadata_dump(ctx, "abcdk");
    if (metadata)
    {
        fprintf(stderr, "%s\n", metadata->pstrs[0]);
        abcdk_object_unref(&metadata);
    }

    abcdk_torch_image_t *img[4] = {0}, *mask[4] = {0};

    // img[0] = abcdk_torch_image_load("/tmp/ccc/you1.jpg", 0);
    // img[3] = abcdk_torch_image_load("/tmp/ccc/you2.jpg", 0);
    // img[1] = abcdk_torch_image_load("/tmp/ccc/you3.jpg", 0);
    // img[2] = abcdk_torch_image_load("/tmp/ccc/you4.jpg", 0);


    img[0] = abcdk_torch_image_load("/home/devel/job/download/eee/1.jpg", 0);
    img[3] = abcdk_torch_image_load("/home/devel/job/download/eee/2.jpg", 0);
    img[1] = abcdk_torch_image_load("/home/devel/job/download/eee/3.jpg", 0);
    img[2] = abcdk_torch_image_load("/home/devel/job/download/eee/4.jpg", 0);

    int chk = abcdk_opencv_stitcher_estimate_transform(ctx, 4, img, mask, 0.8);

    abcdk_opencv_stitcher_build_panorama_param(ctx);

    abcdk_torch_image_t *out = abcdk_torch_image_alloc(ABCDK_TORCH_TAG_HOST);
    chk = abcdk_opencv_stitcher_compose_panorama(ctx, out, 4, img);

    abcdk_torch_image_save("/tmp/ccc/pano.jpg", out);
    abcdk_torch_image_free(&out);

    for (int i = 0; i < 4; i++)
        abcdk_torch_image_free(&img[i]);

    abcdk_opencv_stitcher_destroy(&ctx);

    return 0;
}

int abcdk_test_stitcher_cuda(abcdk_option_t *args)
{
#ifdef __cuda_cuda_h__

    int gpu = abcdk_option_get_int(args, "--gpu", 0, 0);

    CUcontext cuda_ctx = abcdk_cuda_ctx_create(gpu, 0);

    abcdk_cuda_ctx_setspecific(cuda_ctx);
    abcdk_cuda_ctx_push(cuda_ctx);

    abcdk_opencv_stitcher_t *ctx = abcdk_opencv_stitcher_create(ABCDK_TORCH_TAG_HOST);

    abcdk_object_t *metadata = abcdk_object_copyfrom_file(abcdk_option_get(args, "--metadata-load", 0, ""));

    if (metadata)
    {
        abcdk_opencv_stitcher_metadata_load(ctx, "7b1a5e0796419a8278e1f6df640f0bfb", metadata->pstrs[0]);
        abcdk_object_unref(&metadata);
    }

    metadata = abcdk_opencv_stitcher_metadata_dump(ctx, "abcdk");
    if (metadata)
    {
        fprintf(stderr, "%s\n", metadata->pstrs[0]);
        abcdk_object_unref(&metadata);
    }

    abcdk_torch_image_t *cuda_img[4] = {0}, *cuda_mask[4] = {0};
    abcdk_torch_image_t *cpu_img[4] = {0}, *cpu_mask[4] = {0};

    // cpu_img[0] = abcdk_torch_image_load("/tmp/ccc/you1.jpg", 0);
    // cpu_img[3] = abcdk_torch_image_load("/tmp/ccc/you2.jpg", 0);
    // cpu_img[1] = abcdk_torch_image_load("/tmp/ccc/you3.jpg", 0);
    // cpu_img[2] = abcdk_torch_image_load("/tmp/ccc/you4.jpg", 0);

    cpu_img[0] = abcdk_torch_image_load("/home/devel/job/download/eee/1.jpg", 0);
    cpu_img[3] = abcdk_torch_image_load("/home/devel/job/download/eee/2.jpg", 0);
    cpu_img[1] = abcdk_torch_image_load("/home/devel/job/download/eee/3.jpg", 0);
    cpu_img[2] = abcdk_torch_image_load("/home/devel/job/download/eee/4.jpg", 0);

    cuda_img[0] = abcdk_cuda_image_clone(0, cpu_img[0]);
    cuda_img[3] = abcdk_cuda_image_clone(0, cpu_img[3]);
    cuda_img[1] = abcdk_cuda_image_clone(0, cpu_img[1]);
    cuda_img[2] = abcdk_cuda_image_clone(0, cpu_img[2]);

    int chk = abcdk_opencv_stitcher_estimate_transform(ctx, 4, cpu_img, cpu_mask, 0.8);

    abcdk_opencv_stitcher_set_warper(ctx, "spherical");
    // abcdk_opencv_stitcher_set_warper(ctx,"plane");

    abcdk_opencv_stitcher_build_panorama_param(ctx);

    abcdk_torch_image_t *out = abcdk_cuda_image_alloc();

    for (int i = 0; i < 1000; i++)
    {
        chk = abcdk_opencv_stitcher_compose_panorama(ctx, out, 4, cuda_img);

        usleep(40 * 1000);
    }

    abcdk_cuda_image_save("/tmp/ccc/pano.jpg", out);
    abcdk_torch_image_free(&out);

    for (int i = 0; i < 4; i++)
    {
        abcdk_torch_image_free(&cpu_img[i]);
        abcdk_torch_image_free(&cuda_img[i]);
    }

    abcdk_opencv_stitcher_destroy(&ctx);

    abcdk_cuda_ctx_pop();
    abcdk_cuda_ctx_setspecific(NULL);
    abcdk_cuda_ctx_destroy(&cuda_ctx);

#endif //__cuda_cuda_h__

    return 0;
}

int abcdk_test_stitcher(abcdk_option_t *args)
{

    int cmd = abcdk_option_get_int(args, "--cmd", 0, 1);

    if (cmd == 1)
        abcdk_test_stitcher_cpu(args);
    else if (cmd == 2)
        abcdk_test_stitcher_cuda(args);

    return 0;
}
