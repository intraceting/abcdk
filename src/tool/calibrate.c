/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "entry.h"

typedef struct _calibrate
{
    int errcode;
    abcdk_option_t *args;

    abcdk_xpu_context_t *dev_ctx;
    abcdk_xpu_calibrate_t *ctx;

    abcdk_xpu_size_t board_size;
    abcdk_xpu_size_t grid_size;

    int src_num;
    abcdk_xpu_image_t *src_imgs[100];

} calibrate_t;

void _calibrate_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, ABCDK_GETTEXT("\n描述:\n"));

    fprintf(stderr, ABCDK_GETTEXT("\n\t简单的相机标定工具.\n"));

    fprintf(stderr, ABCDK_GETTEXT("\n选项:\n"));

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 显示帮助信息.\n"));

    fprintf(stderr, "\n\t--hwaccel-vendor < VENDOR > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 硬件加速供应商. 默认: %d\n"), ABCDK_XPU_HWACCEL_NONE);
    fprintf(stderr, ABCDK_GETTEXT("\t %d: 无\n"), ABCDK_XPU_HWACCEL_NONE);
    fprintf(stderr, ABCDK_GETTEXT("\t %d: 英伟达\n"), ABCDK_XPU_HWACCEL_NVIDIA);
    fprintf(stderr, ABCDK_GETTEXT("\t %d: 算能\n"), ABCDK_XPU_HWACCEL_SOPHON);
    fprintf(stderr, ABCDK_GETTEXT("\t %d: 瑞芯微\n"), ABCDK_XPU_HWACCEL_ROCKCHIP);

    fprintf(stderr, "\n\t--device-id < ID > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 设备ID. 默认: 0\n"));

    fprintf(stderr, "\n\t--board-size < COLS,ROWS >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 标定板维度(行,列). 默认: 7,11\n"));

    fprintf(stderr, "\n\t--grid-size < HEIGHT,WIDTH >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 网格尺寸(高,宽)(毫米). 默认: 25,25\n"));

    fprintf(stderr, "\n\t--black-alpha < VALUE > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 黑色区域比例(0.0~1.0). 默认: 1\n"));

    fprintf(stderr, "\n\t--src-img-path < PATH >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 源图像路径.默认: ./\n"));

    fprintf(stderr, "\n\t--dst-img-path < PATH >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 矫正图像路径, 未指定则忽略.\n"));

    fprintf(stderr, "\n\t--undistort-param-file < FILE >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 矫正参数文件, 未指定则忽略.\n"));
}

void _calibrate_load_src_img(calibrate_t *ctx)
{
    abcdk_tree_t *dir_ctx = NULL;

    const char *src_img_path_p = abcdk_option_get(ctx->args, "--src-img-path", 0, "./");

    /*set 0.*/
    ctx->src_num = 0;

    abcdk_dirent_open(&dir_ctx, src_img_path_p);

    while (ctx->src_num < ABCDK_ARRAY_SIZE(ctx->src_imgs))
    {
        char file[PATH_MAX] = {0};
        int chk = abcdk_dirent_read(dir_ctx, NULL, file, 1);
        if (chk != 0)
            break;

        ctx->src_imgs[ctx->src_num] = abcdk_xpu_imgcodec_decode_from_file(file);

        if (ctx->src_imgs[ctx->src_num])
        {
            /*统一图像格式. */
            chk = abcdk_xpu_imgproc_convert2(&ctx->src_imgs[ctx->src_num],ABCDK_XPU_PIXFMT_RGB24);
            ctx->src_num += 1;
        }
        else
        {
            abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("加载源图像文件(%s)失败, 无权限或不支持."), file);
        }
    }

    abcdk_tree_free(&dir_ctx);
}

void _calibrate_save_dst_img(calibrate_t *ctx)
{
    abcdk_xpu_image_t *dst = NULL;
    int chk;

    const char *dst_img_path_p = abcdk_option_get(ctx->args, "--dst-img-path", 0, NULL);

    if (!dst_img_path_p)
        return;

    for (int i = 0; i < ctx->src_num; i++)
    {
        char dst_file[PATH_MAX] = {0};

        snprintf(dst_file, PATH_MAX, "%s/undistort-%03d.jpg", dst_img_path_p, i + 1);

        abcdk_xpu_image_t *src_p = ctx->src_imgs[i];

        chk = abcdk_xpu_calibrate_undistort(ctx->ctx, src_p, &dst, ABCDK_XPU_INTER_CUBIC);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("内存不足."));
            break;
        }

        abcdk_xpu_imgcodec_encode_to_file(dst, dst_file, NULL);
    }

    abcdk_xpu_image_free(&dst);
}

void _calibrate_work(calibrate_t *ctx)
{
    int valid_num = 0;
    double rms;
    int chk;

    int hwaccel_vendor = abcdk_option_get_int(ctx->args, "--hwaccel-vendor", 0, ABCDK_XPU_HWACCEL_NONE);
    int device_id = abcdk_option_get_int(ctx->args, "--device-id", 0, 0);

    const char *board_size_p = abcdk_option_get(ctx->args, "--board-size", 0, "7,11");
    const char *grid_size_p = abcdk_option_get(ctx->args, "--grid-size", 0, "25,25");

    double black_alpha = abcdk_option_get_double(ctx->args, "--black-alpha", 0, 1.0);

    const char *undistort_param_file_p = abcdk_option_get(ctx->args, "--undistort-param-file", 0, NULL);

    chk = sscanf(board_size_p, "%d,%d", &ctx->board_size.width, &ctx->board_size.height);
    if (chk != 2 || ctx->board_size.width < 2 || ctx->board_size.height < 2)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("标定板维度(%d >= 2,%d >= 2)错误, 未指定或不支持."), ctx->board_size.width, ctx->board_size.height);
        ABCDK_ERRNO_AND_RETURN0(ctx->errcode = EPERM);
    }

    chk = sscanf(grid_size_p, "%d,%d", &ctx->grid_size.width, &ctx->grid_size.height);
    if (chk != 2)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("网格尺寸(%d >= 5,%d >= 5)错误, 未指定或不支持."), ctx->grid_size.width, ctx->grid_size.height);
        ABCDK_ERRNO_AND_RETURN0(ctx->errcode = EPERM);
    }

    chk = abcdk_xpu_runtime_init(hwaccel_vendor);
    assert(chk == 0);

    ctx->dev_ctx = abcdk_xpu_context_alloc(device_id);
    assert(ctx->dev_ctx != NULL);

    abcdk_xpu_context_current_set(ctx->dev_ctx);

    ctx->ctx = abcdk_xpu_calibrate_alloc();
    assert(ctx->ctx != NULL);

    abcdk_xpu_calibrate_setup(ctx->ctx, ctx->board_size.width, ctx->board_size.height, ctx->grid_size.width, ctx->grid_size.height);

    _calibrate_load_src_img(ctx);

    for (int i = 0; i < ctx->src_num; i++)
    {

        chk = abcdk_xpu_calibrate_detect_corners(ctx->ctx, ctx->src_imgs[i]);
        if (chk < 0)
            continue;

        valid_num += 1;
    }

    if (valid_num < 2)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("源图像(%d)中包含全部角点的有效图像(%d >= 2)太少, 无法评估矫正参数."), ctx->src_num, valid_num);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL,END);
    }

    rms = abcdk_xpu_calibrate_estimate_parameters(ctx->ctx);

    abcdk_trace_printf(LOG_INFO,ABCDK_GETTEXT("重投影误差: %0.6f"),rms);

    chk = abcdk_xpu_calibrate_build_parameters(ctx->ctx, black_alpha);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("内存不足."));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -ENOMEM,END);
    }

    _calibrate_save_dst_img(ctx);


    if(!undistort_param_file_p)
        goto END;

    chk = abcdk_xpu_calibrate_dump_parameters_to_file(ctx->ctx, undistort_param_file_p, "ABCDK");
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("矫正参数写入文件(%s)失败, 无权限或空间不足."), undistort_param_file_p);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -ENOSPC,END);
    }

END:

    for (int i = 0; i < ctx->src_num; i++)
        abcdk_xpu_image_free(&ctx->src_imgs[i]);

    abcdk_xpu_calibrate_free(&ctx->ctx);

    abcdk_xpu_context_unref(&ctx->dev_ctx);
    abcdk_xpu_runtime_deinit();

    return;
}

int abcdk_tool_calibrate(abcdk_option_t *args)
{
    calibrate_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _calibrate_print_usage(ctx.args);
    }
    else
    {
        _calibrate_work(&ctx);
    }

    return ctx.errcode;
}