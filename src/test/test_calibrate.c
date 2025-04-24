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

int _abcdk_test_calibrate_load(abcdk_torch_image_t *img[100], const char *img_path)
{
    abcdk_tree_t *dir_ctx = NULL;
    int count = 0;

    abcdk_dirent_open(&dir_ctx, img_path);

    for (int i = 0; i < 100; i++)
    {
        char file[PATH_MAX] = {0};
        int chk = abcdk_dirent_read(dir_ctx, NULL, file, 1);
        if (chk != 0)
            break;

        img[i] = abcdk_torch_imgcode_load(file);
        assert(img[i]);

        count += 1;
    }

    abcdk_tree_free(&dir_ctx);

    return count;
}

int abcdk_test_calibrate(abcdk_option_t *args)
{

    int cmd = abcdk_option_get_int(args, "--cmd", 0, 1);
    int gpu = abcdk_option_get_int(args, "--gpu", 0, 0);
    const char *board_size_p = abcdk_option_get(args,"--board-size",0,"7,11");
    const char *grid_size_p = abcdk_option_get(args,"--grid-size",0,"25,25");
    const char *img_path_p = abcdk_option_get(args,"--img-path",0,"./");

    abcdk_torch_context_t *torch_ctx = abcdk_torch_context_create(gpu, 0);

    abcdk_torch_context_current_set(torch_ctx);

    abcdk_torch_calibrate_t *ctx = abcdk_torch_calibrate_alloc();

    abcdk_torch_size_t board_size = {-1,-1},grid_size = {-1,-1};

    sscanf(board_size_p,"%d,%d",&board_size.width,&board_size.height);
    sscanf(grid_size_p,"%d,%d",&grid_size.width,&grid_size.height);

    abcdk_torch_calibrate_reset(ctx,&board_size, &grid_size);

    abcdk_torch_image_t *img[100] = {0};

    int count = _abcdk_test_calibrate_load(img,img_path_p);
    int count2 = 0;

    for(int i = 0;i<count;i++)
        count2 = abcdk_torch_calibrate_bind(ctx,img[i]);

    assert(count2 >= 2);

    double rms = abcdk_torch_calibrate_estimate(ctx);

    abcdk_trace_printf(LOG_INFO,"RSM:%0.6f",rms);

    abcdk_torch_image_t *img_p = img[13];
    abcdk_torch_image_t *out = abcdk_torch_image_create(img_p->width,img_p->height,img_p->pixfmt,1);
        
    double camera_matrix[3][3] = {0};
    double dist_coeff[5] = {0};

    abcdk_torch_calibrate_getparam(ctx,camera_matrix, dist_coeff);

    abcdk_torch_image_t *xmap = NULL,* ymap = NULL;
    abcdk_torch_size_t img_size = {img_p->width,img_p->height};

    abcdk_torch_imgproc_undistort_buildmap(&xmap,&ymap,&img_size,0,camera_matrix, dist_coeff);

    for (int i = 0; i < 10000; i++)
        abcdk_torch_imgproc_remap(out, NULL, img_p, NULL, xmap, ymap, 2);

    abcdk_bmp_save_file("/tmp/ccc/img.bmp",img_p->data[0],img_p->stride[0],img_p->width,-img_p->height,24);
    abcdk_bmp_save_file("/tmp/ccc/out.bmp",out->data[0],out->stride[0],out->width,-out->height,24);

    abcdk_torch_imgcode_save("/tmp/ccc/img.jpg", img_p);
    abcdk_torch_imgcode_save("/tmp/ccc/out.jpg", out);

        
    abcdk_torch_image_free(&xmap);
    abcdk_torch_image_free(&ymap);
    
    abcdk_torch_image_free(&out);

    for (int i = 0; i < 6; i++)
        abcdk_torch_image_free(&img[i]);

    abcdk_torch_calibrate_free(&ctx);

    abcdk_torch_context_current_set(NULL);
    abcdk_torch_context_destroy(&torch_ctx);

    return 0;
}