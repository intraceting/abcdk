/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "entry.h"

typedef struct _stitcher
{
    int errcode;
    abcdk_option_t *args;

    abcdk_xpu_context_t *dev_ctx;
    abcdk_xpu_stitcher_t *ctx;

    int src_num;
    abcdk_xpu_image_t *src_imgs[100];

    abcdk_xpu_image_t *dst_img;

} stitcher_t;

void _stitcher_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, ABCDK_GETTEXT("\n描述:\n"));

    fprintf(stderr, ABCDK_GETTEXT("\n\t简单的图像拼接工具.\n"));

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

    fprintf(stderr, "\n\t--feature-name < NAME > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 特征算法. 默认: ORB\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t ORB: Oriented FAST and Rotated BRIEF\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t SIFT: Scale-Invariant Feature Transform\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t SURF: Speeded Up Robust Features\n"));

    fprintf(stderr, "\n\t--warper-name < NAME > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 矫正算法. 默认: spherical\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t plane: 平面\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t spherical: 球面\n"));

    fprintf(stderr, "\n\t--estimate-threshold < VALUE > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 评估阈值(0.1~0.9). 默认: 0.8\n"));

    fprintf(stderr, "\n\t--optimize-seam < BOOL > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 接缝美化. 默认: 1\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t 0: 关\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t 1: 开\n"));

    fprintf(stderr, "\n\t--src-img-path < PATH >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 源图像路径. 默认: ./\n"));

    fprintf(stderr, "\n\t--dst-img-file < FILE >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 全景图像文件(包括路径). 默认: ./panorama.jpg \n"));

    fprintf(stderr, "\n\t--camera-param-file < FILE >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 相机参数文件. 未指定则忽略.\n"));
}

void _stitcher_load_src_img(stitcher_t *ctx)
{
    abcdk_tree_t *dir_ctx = NULL;

    const char *src_img_path_p = abcdk_option_get(ctx->args, "--src-img-path", 0, "./");

    /*set 0.*/
    ctx->src_num = 0;

    abcdk_dirent_open(&dir_ctx, src_img_path_p);

    while (ctx->src_num < 10)
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

void _stitcher_work(stitcher_t *ctx)
{
    int chk;

    int hwaccel_vendor = abcdk_option_get_int(ctx->args, "--hwaccel-vendor", 0, ABCDK_XPU_HWACCEL_NONE);
    int device_id = abcdk_option_get_int(ctx->args, "--device-id", 0, 0);
    const char *feature_name_p = abcdk_option_get(ctx->args, "--feature-name", 0, "ORB");
    const char *warper_name_p = abcdk_option_get(ctx->args, "--warper-name", 0, "spherical");
    float estimate_threshold = abcdk_option_get_double(ctx->args, "--estimate-threshold", 0, 0.8);
    int optimize_seam = abcdk_option_get_int(ctx->args, "--optimize-seam", 0, 1);
    const char *dst_img_file_p = abcdk_option_get(ctx->args, "--dst-img-file", 0, "./panorama.jpg");
    const char *camera_param_file_p = abcdk_option_get(ctx->args, "--camera-param-file", 0, NULL);

    chk = abcdk_xpu_runtime_init(hwaccel_vendor);
    assert(chk == 0);

    ctx->dev_ctx = abcdk_xpu_context_alloc(device_id);
    assert(ctx->dev_ctx != NULL);

    abcdk_xpu_context_current_set(ctx->dev_ctx);

    ctx->ctx = abcdk_xpu_stitcher_alloc();

    chk = abcdk_xpu_stitcher_set_feature_finder(ctx->ctx, feature_name_p);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, "不支持的特征算法(%s).", feature_name_p);
        goto END;
    }

    chk = abcdk_xpu_stitcher_set_warper(ctx->ctx, warper_name_p);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, "不支持的矫正算法(%s).", warper_name_p);
        goto END;
    }

    _stitcher_load_src_img(ctx);
    if(ctx->src_num <2)
    {
        abcdk_trace_printf(LOG_ERR, "至少需要两张图片.");
        goto END;
    }

    chk = abcdk_xpu_stitcher_estimate_parameters(ctx->ctx, ctx->src_num, (const abcdk_xpu_image_t **)ctx->src_imgs, NULL, estimate_threshold);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, "评估相机参数失败, 特征不足或其它错误.");
        goto END;
    }

    chk = abcdk_xpu_stitcher_build_parameters(ctx->ctx);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, "构建相机参数失败, 内存不足或其它错误.");
        goto END;
    }

    chk = abcdk_xpu_stitcher_compose(ctx->ctx, ctx->src_num, (const abcdk_xpu_image_t **)ctx->src_imgs, &ctx->dst_img, optimize_seam);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, "全景拼接失败, 内存不足或其它错误.");
        goto END;
    }

    if (dst_img_file_p)
    {
        chk = abcdk_xpu_imgcodec_encode_to_file(ctx->dst_img, dst_img_file_p, NULL);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_WARNING, "保存全景图像文件(%s)失败, 无空间或无权限.", dst_img_file_p);
        }
    }

    if (camera_param_file_p)
    {
        chk = abcdk_xpu_stitcher_dump_parameters_to_file(ctx->ctx, camera_param_file_p, "ABCDK");
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_WARNING, "保存相机参数文件(%s)失败, 无空间或无权限.", camera_param_file_p);
        }
    }

END:

    for (int i = 0; i < ctx->src_num; i++)
        abcdk_xpu_image_free(&ctx->src_imgs[i]);

    abcdk_xpu_image_free(&ctx->dst_img);

    abcdk_xpu_stitcher_free(&ctx->ctx);

    abcdk_xpu_context_unref(&ctx->dev_ctx);
    abcdk_xpu_runtime_deinit();

    return;
}

int abcdk_tool_stitcher(abcdk_option_t *args)
{
    stitcher_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _stitcher_print_usage(ctx.args);
    }
    else
    {
        _stitcher_work(&ctx);
    }

    return ctx.errcode;
}