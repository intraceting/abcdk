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
    
    int src_count;
    abcdk_xpu_image_t *src_imgs[10];
    int src_start_num[10];
    
    abcdk_xpu_image_t *dst_img;
    int dst_start_num;

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

    fprintf(stderr, "\n\t--camera-param-file < FILE >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 相机参数文件. 未指定则忽略.\n"));

    fprintf(stderr, "\n\t--camera-param-magic < STRING >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 相机参数魔法字符串. 默认: ABCDK \n"));

    fprintf(stderr, "\n\t--src-img-file < FILE >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 源图像文件(包括路径). \n"));

    fprintf(stderr, "\n\t--src-start-num < NUMBER >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 源起始编号. 默认: 1 \n"));

    fprintf(stderr, "\n\t--dst-img-file < FILE >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 全景图像文件(包括路径). 默认: ./panorama.jpg \n"));

    fprintf(stderr, "\n\t--dst-start-num < NUMBER >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 全景始编号. 默认: 1 \n"));
}

int _stitcher_fetch_src_img(stitcher_t *ctx, int64_t pts)
{
    int chk;

    static const char *src_file_p[10] = {NULL};

    if (ctx->src_count <= 0)
    {
        for (int i = 0; i < ABCDK_ARRAY_SIZE(src_file_p); i++)
        {
            src_file_p[i] = abcdk_option_get(ctx->args, "--src-img-file", i, NULL);
            if (!src_file_p[i])
                break;

            ctx->src_start_num[i] = abcdk_option_get_int(ctx->args, "--src-start-num", i, 1);

            ctx->src_count += 1;
        }
    }

    for (int i = 0; i < ctx->src_count; i++)
    {
        char pathfile[PATH_MAX] = {0};
        snprintf(pathfile, PATH_MAX, src_file_p[i], ctx->src_start_num[i]++);

        if (pts > 0)
        {
            if (abcdk_strcmp(pathfile, src_file_p[i], 1) == 0)
                return -ENOENT;
        }

        abcdk_xpu_image_free(&ctx->src_imgs[i]);//free old.
        ctx->src_imgs[i] = abcdk_xpu_imgcodec_decode_from_file(pathfile);
        if (ctx->src_imgs[i])
        {
            /*统一图像格式. */
            chk = abcdk_xpu_imgproc_convert2(&ctx->src_imgs[i], ABCDK_XPU_PIXFMT_RGB24);
            if(chk != 0)
                return -EINVAL;
        }
        else
        {
            abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("加载源图像文件(%s)失败, 不存在或无权限."), pathfile);
            return -ENOENT;
        }
    }

    return 0;
}

int _stitcher_dump_dst_img(stitcher_t *ctx)
{
    static const char *dst_img_file_p = NULL;
    int chk;
    
    if(!dst_img_file_p)
    {
        dst_img_file_p = abcdk_option_get(ctx->args, "--dst-img-file", 0, "./panorama.jpg");
        ctx->dst_start_num = abcdk_option_get_int(ctx->args, "--dst-start-num", 0, 1);
    }

    if(!dst_img_file_p || !*dst_img_file_p)
        return 0;
    
    char pathfile[PATH_MAX] = {0};
    snprintf(pathfile, PATH_MAX, dst_img_file_p, ctx->dst_start_num++);

    chk = abcdk_xpu_imgcodec_encode_to_file(ctx->dst_img, pathfile, NULL);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("保存全景图像文件(%s)失败, 无空间或无权限."), pathfile);
        return -ENOSPC;
    }

    return 0;
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
    const char *camera_param_file_p = abcdk_option_get(ctx->args, "--camera-param-file", 0, NULL);
    const char *camera_param_magic_p = abcdk_option_get(ctx->args, "--camera-param-magic", 0, "ABCDK");

    chk = abcdk_xpu_runtime_init(hwaccel_vendor);
    assert(chk == 0);

    ctx->dev_ctx = abcdk_xpu_context_alloc(device_id);
    assert(ctx->dev_ctx != NULL);

    abcdk_xpu_context_current_set(ctx->dev_ctx);

    chk = _stitcher_fetch_src_img(ctx, 0);
    if(chk != 0 || ctx->src_count <2)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("源图像至少需要两张."));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL,END);
    }

    ctx->ctx = abcdk_xpu_stitcher_alloc();

    if(camera_param_file_p && access(camera_param_file_p,R_OK) == 0)
    {
        chk = abcdk_xpu_stitcher_load_parameters_from_file(ctx->ctx, camera_param_file_p, camera_param_magic_p);
        if (chk == -127)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("加载相机参数文件(%s)成功, 但与当前源图像不匹配."), camera_param_file_p);
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL,END);
        }
        else if (chk <0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("加载相机参数文件(%s)失败, 格式错误或无权限."), camera_param_file_p);
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL,END);
        }
    }
    else
    {
        chk = abcdk_xpu_stitcher_set_feature_finder(ctx->ctx, feature_name_p);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("不支持的特征算法(%s)."), feature_name_p);
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL,END);
        }

        chk = abcdk_xpu_stitcher_estimate_parameters(ctx->ctx, ctx->src_count, (const abcdk_xpu_image_t **)ctx->src_imgs, NULL, estimate_threshold);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("评估相机参数失败, 特征不足或其它错误."));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL,END);
        }

        if (camera_param_file_p)
        {
            chk = abcdk_xpu_stitcher_dump_parameters_to_file(ctx->ctx, camera_param_file_p, camera_param_magic_p);
            if (chk != 0)
            {
                abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("保存相机参数文件(%s)失败, 无空间或无权限. 忽略."), camera_param_file_p);
            }
        }
    }

    chk = abcdk_xpu_stitcher_set_warper(ctx->ctx, warper_name_p);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("不支持的矫正算法(%s)."), warper_name_p);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL,END);
    }

    chk = abcdk_xpu_stitcher_build_parameters(ctx->ctx);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("构建相机参数失败, 内存不足或其它错误."));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL,END);
    }

    chk = abcdk_xpu_stitcher_compose(ctx->ctx, ctx->src_count, (const abcdk_xpu_image_t **)ctx->src_imgs, &ctx->dst_img, optimize_seam);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("全景拼接失败, 内存不足或其它错误."));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -ENOMEM,END);
    }

    chk = _stitcher_dump_dst_img(ctx);
    if(chk != 0)
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = chk,END);

    for (int64_t i = 1; i < INT64_MAX; i++)
    {
        chk = _stitcher_fetch_src_img(ctx, i);
        if (chk != 0)
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = 0, END); // no error.

        chk = abcdk_xpu_stitcher_compose(ctx->ctx, ctx->src_count, (const abcdk_xpu_image_t **)ctx->src_imgs, &ctx->dst_img, optimize_seam);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("全景拼接失败, 内存不足或其它错误."));
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -ENOMEM, END);
        }

        chk = _stitcher_dump_dst_img(ctx);
        if (chk != 0)
            ABCDK_ERRNO_AND_GOTO1(ctx->errcode = chk, END);

        abcdk_trace_printf(LOG_DEBUG, ABCDK_GETTEXT("已处理: %lld"), i);
    }

END:

    for (int i = 0; i < ctx->src_count; i++)
        abcdk_xpu_image_free(&ctx->src_imgs[i]);

    abcdk_xpu_image_free(&ctx->dst_img);

    abcdk_xpu_stitcher_free(&ctx->ctx);

    abcdk_xpu_context_current_set(NULL);
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