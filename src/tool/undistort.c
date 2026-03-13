/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "entry.h"

typedef struct _undistort
{
    int errcode;
    abcdk_option_t *args;

    abcdk_xpu_context_t *dev_ctx;
    abcdk_xpu_calibrate_t *ctx;
} undistort_t;

void _undistort_print_usage(abcdk_option_t *args)
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

    fprintf(stderr, "\n\t--undistort-black-alpha < VALUE > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 黑色区域比例(0.0~1.0). 默认: 0\n"));
    
    fprintf(stderr, "\n\t--undistort-param-magic < STRING >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 矫正参数魔法字符串. 默认: ABCDK \n"));

    fprintf(stderr, "\n\t--undistort-param-file < FILE >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 矫正参数文件.\n"));

    fprintf(stderr, "\n\t--src-img-path < PATH >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 源图像路径. 默认: ./\n"));

    fprintf(stderr, "\n\t--dst-img-path < PATH >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 矫正图像路径. 默认: ./out/\n"));
}

void _undistort_process(undistort_t *ctx)
{
    abcdk_tree_t *dir_ctx = NULL;
    abcdk_xpu_image_t *src_img = NULL;
    abcdk_xpu_image_t *dst_img = NULL;

    const char *src_img_path_p = abcdk_option_get(ctx->args, "--src-img-path", 0, "./");
    const char *dst_img_path_p = abcdk_option_get(ctx->args, "--dst-img-path", 0, "./out/");
 
    /*创建可能不存在的路径.*/
    abcdk_mkdir(dst_img_path_p, 0755);

    abcdk_dirent_open(&dir_ctx, src_img_path_p);

    for (int64_t i = 1; i < INT64_MAX; i++)
    {
        char file[PATH_MAX] = {0};
        int chk = abcdk_dirent_read(dir_ctx, NULL, file, 1);
        if (chk != 0)
            break;

        abcdk_xpu_image_free(&src_img);
        src_img = abcdk_xpu_imgcodec_decode_from_file(file);
        if (!src_img)
        {
            abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("加载源图像文件(%s)失败, 无权限或不支持. 忽略."), file);
            continue;
        }

        /**统一图像格式. */
        chk = abcdk_xpu_imgproc_convert2(&src_img, ABCDK_XPU_PIXFMT_RGB24);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("转换图像格式失败, 内存不足或其它."));
            break;
        }

        chk = abcdk_xpu_calibrate_undistort(ctx->ctx, src_img, &dst_img, ABCDK_XPU_INTER_CUBIC);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("内存不足."));
            break;
        }

        char out_name[NAME_MAX] = {0};
        char out_file[PATH_MAX] = {0};

        abcdk_basename(out_name, file);
        sprintf(out_file, "%s/%s", dst_img_path_p, out_name);

        chk = abcdk_xpu_imgcodec_encode_to_file(dst_img, out_file, NULL);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("保存矫正图像文件(%s)失败, 无空间或无权限."), out_file);
            break;
        }

        abcdk_trace_printf(LOG_DEBUG, ABCDK_GETTEXT("已处理: %lld"), i);
    }

    abcdk_xpu_image_free(&src_img);
    abcdk_xpu_image_free(&dst_img);
    abcdk_tree_free(&dir_ctx);
}

void _undistort_work(undistort_t *ctx)
{
    int chk;

    int hwaccel_vendor = abcdk_option_get_int(ctx->args, "--hwaccel-vendor", 0, ABCDK_XPU_HWACCEL_NONE);
    int device_id = abcdk_option_get_int(ctx->args, "--device-id", 0, 0);

    double black_alpha = abcdk_option_get_double(ctx->args, "--undistort-black-alpha", 0, 0.);

    const char *undistort_param_file_p = abcdk_option_get(ctx->args, "--undistort-param-file", 0, NULL);
    const char *undistort_param_magic_p = abcdk_option_get(ctx->args, "--undistort-param-magic", 0, "ABCDK");


    chk = abcdk_xpu_runtime_init(hwaccel_vendor);
    assert(chk == 0);

    ctx->dev_ctx = abcdk_xpu_context_alloc(device_id);
    assert(ctx->dev_ctx != NULL);

    abcdk_xpu_context_current_set(ctx->dev_ctx);

    ctx->ctx = abcdk_xpu_calibrate_alloc();
    assert(ctx->ctx != NULL);

    chk = abcdk_xpu_calibrate_load_parameters_from_file(ctx->ctx, undistort_param_file_p, undistort_param_magic_p);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("加载标定参数文件(%s)失败, 不存在或无权限."), undistort_param_file_p);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EPERM,END);
    }

    chk = abcdk_xpu_calibrate_build_parameters(ctx->ctx, black_alpha);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("内存不足."));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -ENOMEM,END);
    }

    _undistort_process(ctx);

END:

    abcdk_xpu_calibrate_free(&ctx->ctx);

    abcdk_xpu_context_unref(&ctx->dev_ctx);
    abcdk_xpu_runtime_deinit();

    return;
}

int abcdk_tool_undistort(abcdk_option_t *args)
{
    undistort_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _undistort_print_usage(ctx.args);
    }
    else
    {
        _undistort_work(&ctx);
    }

    return ctx.errcode;
}