/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "entry.h"

typedef struct _abcdk_calibrate
{
    int errcode;
    abcdk_option_t *args;

    abcdk_torch_context_t *torch_ctx;
    abcdk_torch_calibrate_t *calibrate_ctx;

    abcdk_torch_size_t board_size;
    abcdk_torch_size_t grid_size;
    int bind_num;
    double estimate_rms;
    double camera_matrix[3][3];
    double dist_coeff[5];
    abcdk_torch_size_t camera_size;

    abcdk_torch_image_t *remap_xmap;
    abcdk_torch_image_t *remap_ymap;

    abcdk_torch_image_t *src_imgs[100];
    int src_num;


}abcdk_calibrate_t;

void _abcdk_calibrate_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, "\n描述:\n");

    fprintf(stderr, "\n\t简单的相机标定工具。\n");

    fprintf(stderr, "\n选项:\n");

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, "\t\t显示帮助信息。\n");

    fprintf(stderr, "\n\t--board-size < COLS,ROWS >\n");
    fprintf(stderr, "\t\t标定板维度(行,列)。默认：7,11\n");

    fprintf(stderr, "\n\t--grid-size < HEIGHT,WIDTH >\n");
    fprintf(stderr, "\t\t网格尺寸(高,宽)(毫米)。默认：25,25\n");

    fprintf(stderr, "\n\t--src-img-path < PATH >\n");
    fprintf(stderr, "\t\t源图像路径。默认：./\n");

    fprintf(stderr, "\n\t--dst-img-path < PATH >\n");
    fprintf(stderr, "\t\t矫正图像路径，未指定则忽略。\n");

    fprintf(stderr, "\n\t--undistort-param-file < FILE >\n");
    fprintf(stderr, "\t\t矫正参数文件，未指定则忽略。\n");
}

void _abcdk_calibrate_load_src_img(abcdk_calibrate_t *ctx)
{
    abcdk_tree_t *dir_ctx = NULL;

    const char *src_img_path_p = abcdk_option_get(ctx->args,"--src-img-path",0,"./");

    /*set 0.*/
    ctx->src_num = 0;

    abcdk_dirent_open(&dir_ctx, src_img_path_p);

    while(ctx->src_num < 100)
    {
        char file[PATH_MAX] = {0};
        int chk = abcdk_dirent_read(dir_ctx, NULL, file, 1);
        if (chk != 0)
            break;

        ctx->src_imgs[ctx->src_num] = abcdk_torch_imgcode_load_host(file);
        if(!ctx->src_imgs[ctx->src_num])
        {
            abcdk_trace_printf(LOG_WARNING,"加载源图像文件(%s)失败，无权限或不支持。",file);
        }
        else 
        {
            ctx->src_num += 1;
        }
    }

    abcdk_tree_free(&dir_ctx);
}

void _abcdk_calibrate_save_dst_img(abcdk_calibrate_t *ctx)
{
    abcdk_torch_image_t *dst = NULL;

    const char *dst_img_path_p = abcdk_option_get(ctx->args, "--dst-img-path", 0, NULL);

    if(!dst_img_path_p)
        return;
    
    abcdk_torch_imgproc_undistort_buildmap_host(&ctx->remap_xmap, &ctx->remap_ymap, &ctx->camera_size, 0, ctx->camera_matrix, ctx->dist_coeff);

    for (int i = 0; i < ctx->src_num; i++)
    {
        char dst_file[PATH_MAX] = {0};

        snprintf(dst_file, PATH_MAX, "%s/undistort-%03d.jpg", dst_img_path_p, i + 1);

        abcdk_torch_image_t *src_p = ctx->src_imgs[i];

        abcdk_torch_image_reset_host(&dst,src_p->width,src_p->height,src_p->pixfmt,1);
        if(!dst)
        {
            abcdk_trace_printf(LOG_ERR,"内存不足。");
            break;
        }

        abcdk_torch_imgproc_remap_host(dst, NULL, src_p, NULL, ctx->remap_xmap, ctx->remap_ymap, ABCDK_TORCH_INTER_CUBIC);

        abcdk_torch_imgcode_save_host(dst_file, dst);
    }

    abcdk_torch_image_free_host(&dst);

    abcdk_torch_image_free_host(&ctx->remap_xmap);
    abcdk_torch_image_free_host(&ctx->remap_ymap);
}

void _abcdk_calibrate_work(abcdk_calibrate_t *ctx)
{
    int chk;

    const char *board_size_p = abcdk_option_get(ctx->args,"--board-size",0,"7,11");
    const char *grid_size_p = abcdk_option_get(ctx->args,"--grid-size",0,"25,25");
    const char *undistort_param_file_p = abcdk_option_get(ctx->args,"--undistort-param-file",0, NULL);

    chk = sscanf(board_size_p, "%d,%d", &ctx->board_size.width, &ctx->board_size.height);
    if (chk != 2 || ctx->board_size.width < 2 || ctx->board_size.height < 2)
    {
        abcdk_trace_printf(LOG_ERR,"标定板维度(%d >= 2,%d >= 2)错误，未指定或不支持。",ctx->board_size.width, ctx->board_size.height);
        ABCDK_ERRNO_AND_RETURN0(ctx->errcode = EPERM);
    }

    chk = sscanf(grid_size_p,"%d,%d",&ctx->grid_size.width,&ctx->grid_size.height);
    if(chk != 2)
    {
        abcdk_trace_printf(LOG_ERR,"网格尺寸(%d >= 5,%d >= 5)错误，未指定或不支持。",ctx->grid_size.width, ctx->grid_size.height);
        ABCDK_ERRNO_AND_RETURN0(ctx->errcode = EPERM);
    }

    ctx->torch_ctx = abcdk_torch_context_create_host(0, 0);
    assert(ctx->torch_ctx != NULL);

    chk = abcdk_torch_context_current_set_host(ctx->torch_ctx);
    assert(chk == 0);

    ctx->calibrate_ctx = abcdk_torch_calibrate_alloc_host();
    assert(ctx->calibrate_ctx != NULL);

    chk = abcdk_torch_calibrate_reset_host(ctx->calibrate_ctx,&ctx->board_size, &ctx->grid_size);
    assert(chk == 0);

    _abcdk_calibrate_load_src_img(ctx);

    for (int i = 0; i < ctx->src_num; i++)
        ctx->bind_num = abcdk_torch_calibrate_bind_host(ctx->calibrate_ctx, ctx->src_imgs[i]);

    if (ctx->bind_num < 2)
    {
        abcdk_trace_printf(LOG_ERR,"源图像(%d)中包含全部角点的有效图像(%d >= 2)太少，无法评估矫正参数。",ctx->src_num,ctx->bind_num);
        goto END;
    }

    ctx->estimate_rms = abcdk_torch_calibrate_estimate_host(ctx->calibrate_ctx);

    abcdk_trace_printf(LOG_INFO,"RSM: %0.6f",ctx->estimate_rms);

    chk = abcdk_torch_calibrate_getparam_host(ctx->calibrate_ctx,ctx->camera_matrix, ctx->dist_coeff);
    assert(chk == 0);

    ctx->camera_size.width = ctx->src_imgs[0]->width;
    ctx->camera_size.height = ctx->src_imgs[0]->height;

    abcdk_trace_printf(LOG_INFO,"----------camera-size----------");
    abcdk_trace_printf(LOG_INFO,"width= %d",ctx->camera_size.width);
    abcdk_trace_printf(LOG_INFO,"height= %d",ctx->camera_size.height);
    abcdk_trace_printf(LOG_INFO,"----------camera-size----------");

    abcdk_trace_printf(LOG_INFO,"----------camera-matrix----------");
    abcdk_trace_printf(LOG_INFO,"fx= %.09lf",ctx->camera_matrix[0][0]);
    abcdk_trace_printf(LOG_INFO,"fy= %.09lf",ctx->camera_matrix[1][1]);
    abcdk_trace_printf(LOG_INFO,"cx= %.09lf",ctx->camera_matrix[0][2]);
    abcdk_trace_printf(LOG_INFO,"cy= %.09lf",ctx->camera_matrix[1][2]);
    abcdk_trace_printf(LOG_INFO,"----------camera-matrix----------");

    abcdk_trace_printf(LOG_INFO,"----------dist-coeff----------");
    abcdk_trace_printf(LOG_INFO,"k1= %.09lf",ctx->dist_coeff[0]);
    abcdk_trace_printf(LOG_INFO,"k2= %.09lf",ctx->dist_coeff[1]);
    abcdk_trace_printf(LOG_INFO,"p1= %.09lf",ctx->dist_coeff[2]);
    abcdk_trace_printf(LOG_INFO,"p2= %.09lf",ctx->dist_coeff[3]);
    abcdk_trace_printf(LOG_INFO,"k3= %.09lf",ctx->dist_coeff[4]);
    abcdk_trace_printf(LOG_INFO,"----------dist-coeff----------");

    if(undistort_param_file_p)
    {
        chk = abcdk_torch_calibrate_param_dump_file(undistort_param_file_p,&ctx->camera_size,ctx->camera_matrix,ctx->dist_coeff);
        if(chk != 0)
        {
            abcdk_trace_printf(LOG_ERR,"矫正参数写入文件(%s)失败，无权限或空间不足。", undistort_param_file_p);
            goto END;
        }
    }


    _abcdk_calibrate_save_dst_img(ctx);

END:

    for (int i = 0; i < ctx->src_num; i++)
        abcdk_torch_image_free_host(&ctx->src_imgs[i]);

    abcdk_torch_calibrate_free_host(&ctx->calibrate_ctx);

    abcdk_torch_context_current_set_host(NULL);
    abcdk_torch_context_destroy_host(&ctx->torch_ctx);

    return;
}

int abcdk_tool_calibrate(abcdk_option_t *args)
{
    abcdk_calibrate_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _abcdk_calibrate_print_usage(ctx.args);
    }
    else
    {
        _abcdk_calibrate_work(&ctx);
    }

    return ctx.errcode;
}