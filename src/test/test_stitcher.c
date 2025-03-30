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

int abcdk_test_stitcher(abcdk_option_t *args)
{

    int cmd = abcdk_option_get_int(args, "--cmd", 0, 1);
    int gpu = abcdk_option_get_int(args, "--gpu", 0, 0);

    abcdk_torch_context_t *torch_ctx = abcdk_torch_context_create(gpu, 0);

    abcdk_torch_context_current_set(torch_ctx);

    abcdk_torch_stitcher_t *ctx = abcdk_torch_stitcher_create();

    abcdk_object_t *metadata = abcdk_object_copyfrom_file(abcdk_option_get(args, "--metadata-load", 0, ""));

    if (metadata)
    {
        abcdk_torch_stitcher_metadata_load(ctx, "7b1a5e0796419a8278e1f6df640f0bfb", metadata->pstrs[0]);
        abcdk_object_unref(&metadata);
    }

    metadata = abcdk_torch_stitcher_metadata_dump(ctx, "abcdk");
    if (metadata)
    {
        fprintf(stderr, "%s\n", metadata->pstrs[0]);
        abcdk_object_unref(&metadata);
    }

    abcdk_torch_image_t *img[6] = {0}, *mask[6] = {0};

    // img[0] = abcdk_torch_imgcode_load("/tmp/ccc/you1.jpg", 0);
    // img[3] = abcdk_torch_imgcode_load("/tmp/ccc/you2.jpg", 0);
    // img[1] = abcdk_torch_imgcode_load("/tmp/ccc/you3.jpg", 0);
    // img[2] = abcdk_torch_imgcode_load("/tmp/ccc/you4.jpg", 0);

    img[0] = abcdk_torch_imgcode_load("/home/devel/job/download/eee/1.jpg");
    img[3] = abcdk_torch_imgcode_load("/home/devel/job/download/eee/2.jpg");
    img[1] = abcdk_torch_imgcode_load("/home/devel/job/download/eee/3.jpg");
    img[2] = abcdk_torch_imgcode_load("/home/devel/job/download/eee/4.jpg");
    img[5] = abcdk_torch_imgcode_load("/home/devel/job/download/eee/5.jpg");
    img[4] = abcdk_torch_imgcode_load("/home/devel/job/download/eee/6.jpg");

    //abcdk_torch_imgcode_save("/tmp/ccc/a.jpg",img[0]);

    // abcdk_opencv_stitcher_set_feature(ctx,"SURF");
    abcdk_torch_stitcher_set_feature(ctx, "SIFT");

    int chk = abcdk_torch_stitcher_estimate(ctx, 6, img, mask, 0.8);

    // abcdk_opencv_stitcher_set_warper(ctx,"plane");
    abcdk_torch_stitcher_build(ctx);

    abcdk_torch_image_t *out = abcdk_torch_image_alloc();
    chk = abcdk_torch_stitcher_compose(ctx, out, 6, img);

    abcdk_torch_imgcode_save("/tmp/ccc/pano.jpg", out);
    abcdk_torch_image_free(&out);

    for (int i = 0; i < 6; i++)
        abcdk_torch_image_free(&img[i]);

    abcdk_torch_stitcher_destroy(&ctx);

    abcdk_torch_context_current_set(NULL);
    abcdk_torch_context_destroy(&torch_ctx);

    return 0;
}
