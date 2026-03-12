/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "entry.h"

typedef struct _videostitcher
{
    int errcode;
    abcdk_option_t *args;

    abcdk_xpu_context_t *dev_ctx;
    abcdk_xpu_stitcher_t *ctx;

    int optimize_seam;

    int src_count;
    const char *src_file_p[10];

    const char *dst_file_p;
    int dst_fps;

    int64_t src_pts[10];
    abcdk_xpu_image_t *src_img[10];
    abcdk_rwlock_t *src_locker[10];
    abcdk_xpu_context_t *src_dev_ctx[10];

    int64_t dst_dts;
    abcdk_xpu_image_t *dst_img;
    abcdk_rwlock_t *dst_locker;
    abcdk_xpu_context_t *dst_dev_ctx;

    abcdk_worker_t *worker_ctx;
    int worker_exit_flag;

} videostitcher_t;

void _videostitcher_print_usage(abcdk_option_t *args)
{
    fprintf(stderr, ABCDK_GETTEXT("\n描述:\n"));

    fprintf(stderr, ABCDK_GETTEXT("\n\t简单的视频图像拼接工具.\n"));

    fprintf(stderr, ABCDK_GETTEXT("\n选项:\n"));

    fprintf(stderr, "\n\t--help\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 显示帮助信息.\n"));

    fprintf(stderr, "\n\t--hwaccel-vendor < VENDOR > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 硬件加速供应商. 默认: %d\n"), ABCDK_XPU_HWACCEL_NONE);
    fprintf(stderr, ABCDK_GETTEXT("\t %d: 无\n"), ABCDK_XPU_HWACCEL_NONE);
    fprintf(stderr, ABCDK_GETTEXT("\t %d: 英伟达\n"), ABCDK_XPU_HWACCEL_NVIDIA);

    fprintf(stderr, "\n\t--device-id < ID > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 设备ID. 默认: 0\n"));

    fprintf(stderr, "\n\t--warper-name < NAME > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 矫正算法. 默认: spherical\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t plane: 平面\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t spherical: 球面\n"));

    fprintf(stderr, "\n\t--optimize-seam < BOOL > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 接缝美化. 默认: 1\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t 0: 关\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t 1: 开\n"));

    fprintf(stderr, "\n\t--camera-param-file < FILE >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 相机参数文件. 未指定则忽略.\n"));

    fprintf(stderr, "\n\t--camera-param-magic < STRING >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 相机参数魔法字符串. 默认: ABCDK \n"));

    fprintf(stderr, "\n\t--src-video-file < FILE >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 源视频文件(包括路径). \n"));

    fprintf(stderr, "\n\t--dst-video-file < FILE >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 全景视频文件(包括路径). 默认: ./panorama.mp4 \n"));

    fprintf(stderr, "\n\t--dst-video-fps < NUMBER >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 全景视频帧率. 默认: 25 \n"));
}

void _videostitcher_reader(videostitcher_t *ctx, int id)
{
    abcdk_xpu_vdec_t *dec_ctx = NULL;
    abcdk_ffmpeg_editor_t *ff_ctx = NULL;
    abcdk_ffmpeg_editor_param_t ff_param = {0};
    AVPacket *ff_pkt = av_packet_alloc();
    int chk;

    abcdk_xpu_context_current_set(ctx->src_dev_ctx[id]);

    ff_param.url = ctx->src_file_p[id];
    ff_param.timeout = 5;
    ff_param.read_mp4toannexb = 1;
    ff_param.read_ignore_audio = 1;
    ff_param.read_ignore_subtitle = 1;
    ff_param.read_nodelay = 0;

    for (int i = 0; i < 100000000; i++)
    {
        if (abcdk_atomic_compare(&ctx->worker_exit_flag, 1))
            break;

        if (i > 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("原视频文件[%d](%s)到末尾或已断开, 2秒稍后重试."), id, ctx->src_file_p[id]);
            sleep(2);
        }

        abcdk_xpu_vdec_free(&dec_ctx);
        abcdk_ffmpeg_editor_free(&ff_ctx);

        dec_ctx = abcdk_xpu_vdec_alloc();
        ff_ctx = abcdk_ffmpeg_editor_alloc(0);

        chk = abcdk_ffmpeg_editor_open(ff_ctx, &ff_param);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("原视频文件[%d](%s打开失败, 不存在或无权限."), id, ctx->src_file_p[id]);
            continue;
        }

        for (int j = 0; j < abcdk_ffmpeg_editor_stream_nb(ff_ctx); j++)
        {
            AVStream *p = abcdk_ffmpeg_editor_stream_ctx(ff_ctx, j);

            if (p->codecpar->codec_type != AVMEDIA_TYPE_VIDEO)
                continue;

            abcdk_xpu_vcodec_params_t dec_params = {0};

            if (p->codecpar->codec_id == AV_CODEC_ID_H264)
                dec_params.format = ABCDK_XPU_VCODEC_ID_H264;
            if (p->codecpar->codec_id == AV_CODEC_ID_H265)
                dec_params.format = ABCDK_XPU_VCODEC_ID_H265;

            dec_params.ext_data = p->codecpar->extradata;
            dec_params.ext_size = p->codecpar->extradata_size;

            chk = abcdk_xpu_vdec_setup(dec_ctx, &dec_params);
            assert(chk == 0);
            break;
        }

        while (1)
        {
            if (abcdk_atomic_compare(&ctx->worker_exit_flag, 1))
                break;

            chk = abcdk_ffmpeg_editor_read_packet(ff_ctx, ff_pkt);
            if (chk != 0)
                break;

            abcdk_rwlock_wrlock(ctx->src_locker[id], 1); // lock
            /*先从解码器中取出图像, 以便释放解码空间.*/
            chk = abcdk_xpu_vdec_recv_frame(dec_ctx, &ctx->src_img[id], &ctx->src_pts[id]);
            /*统一图像格式.*/
            if (chk > 0)
                chk = abcdk_xpu_imgproc_convert2(&ctx->src_img[id], ABCDK_XPU_PIXFMT_RGB24);
            abcdk_rwlock_unlock(ctx->src_locker[id]); // unlock.

            if (chk < 0)
                break;

            /*向解码器发送数据包.*/
            chk = abcdk_xpu_vdec_send_packet(dec_ctx, ff_pkt->data, ff_pkt->size, ff_pkt->pts);
            if (chk < 0)
                break;

            abcdk_trace_printf(LOG_DEBUG, "SRC[%d],PTS[%.3f]", id, abcdk_ffmpeg_editor_stream_ts2sec(ff_ctx, ff_pkt->stream_index, ctx->src_pts[id]));
        }
    }

    abcdk_xpu_vdec_free(&dec_ctx);
    abcdk_ffmpeg_editor_free(&ff_ctx);
    av_packet_free(&ff_pkt);

    abcdk_xpu_context_current_set(NULL);
    abcdk_xpu_context_unref(&ctx->src_dev_ctx[id]);
}

void _videostitcher_stitching(videostitcher_t *ctx)
{
    int src_count;
    int64_t src_pts[10] = {INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN};
    abcdk_xpu_image_t *src_img[10] = {NULL};
    int chk;

    abcdk_xpu_context_current_set(ctx->dst_dev_ctx);

    while (1)
    {
        usleep(10 * 1000); // 100FPS.
        
        for (int i = 0; i < ctx->src_count; i++)
        {
            abcdk_rwlock_rdlock(ctx->src_locker[i], 1);

            if (ctx->src_img[i] && !abcdk_xpu_image_empty(ctx->src_img[i]))
            {
                src_pts[i] = ctx->src_pts[i]; // copy

                int w = abcdk_xpu_image_get_width(ctx->src_img[i]);
                int h = abcdk_xpu_image_get_height(ctx->src_img[i]);
                abcdk_xpu_pixfmt_t f = abcdk_xpu_image_get_pixfmt(ctx->src_img[i]);

                abcdk_xpu_image_reset(&src_img[i], w, h, f, 16);
                abcdk_xpu_image_copy(ctx->src_img[i], src_img[i]);
            }

            abcdk_rwlock_unlock(ctx->src_locker[i]);
        }

        src_count = 0; // set to 0.
        for (int i = 0; i < ctx->src_count; i++)
        {
            if (src_img[i] && !abcdk_xpu_image_empty(src_img[i]))
                src_count += 1;
        }

        if (src_count != ctx->src_count)
            continue;

        abcdk_rwlock_wrlock(ctx->dst_locker, 1);

        /*拼接.*/
        chk = abcdk_xpu_stitcher_compose(ctx->ctx, src_count, (const abcdk_xpu_image_t **)src_img, &ctx->dst_img, ctx->optimize_seam);
        /*滚动DTS.*/
        if (chk == 0)
            ctx->dst_dts += 1;

        abcdk_rwlock_unlock(ctx->dst_locker);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("全景拼接失败, 内存不足或其它错误."));
            break;
        }

        abcdk_trace_printf(LOG_DEBUG, "STITCHING,DTS[%lld]", ctx->dst_dts);
    }

    abcdk_xpu_context_current_set(NULL);
    abcdk_xpu_context_unref(&ctx->dst_dev_ctx);
}

void _videostitcher_writer(videostitcher_t *ctx)
{
}

void _videostitcher_process_cb(void *opaque, uint64_t event, void *item)
{
    videostitcher_t *ctx = (videostitcher_t *)opaque;

    if (event < 100)
        _videostitcher_reader(ctx, event);
    else if (event == 101)
        _videostitcher_stitching(ctx);
    else if (event == 102)
        _videostitcher_writer(ctx);
}

void _videostitcher_work(videostitcher_t *ctx)
{
    int chk;

    int hwaccel_vendor = abcdk_option_get_int(ctx->args, "--hwaccel-vendor", 0, ABCDK_XPU_HWACCEL_NONE);
    int device_id = abcdk_option_get_int(ctx->args, "--device-id", 0, 0);
    const char *warper_name_p = abcdk_option_get(ctx->args, "--warper-name", 0, "spherical");
    int optimize_seam = abcdk_option_get_int(ctx->args, "--optimize-seam", 0, 1);
    const char *camera_param_file_p = abcdk_option_get(ctx->args, "--camera-param-file", 0, NULL);
    const char *camera_param_magic_p = abcdk_option_get(ctx->args, "--camera-param-magic", 0, "ABCDK");

    chk = abcdk_xpu_runtime_init(hwaccel_vendor);
    assert(chk == 0);

    ctx->dev_ctx = abcdk_xpu_context_alloc(device_id);
    assert(ctx->dev_ctx != NULL);

    abcdk_xpu_context_current_set(ctx->dev_ctx);

    ctx->ctx = abcdk_xpu_stitcher_alloc();

    if (!camera_param_file_p || access(camera_param_file_p, R_OK) != 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("相机参数文件(%s)不存在或无权限."), camera_param_file_p);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL, END);
    }

    chk = abcdk_xpu_stitcher_load_parameters_from_file(ctx->ctx, camera_param_file_p, camera_param_magic_p);
    if (chk == -127)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("加载相机参数文件(%s)成功, 但与当前源图像不匹配."), camera_param_file_p);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL, END);
    }
    else if (chk < 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("加载相机参数文件(%s)失败, 格式错误或无权限."), camera_param_file_p);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL, END);
    }

    chk = abcdk_xpu_stitcher_set_warper(ctx->ctx, warper_name_p);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("不支持的矫正算法(%s)."), warper_name_p);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL, END);
    }

    chk = abcdk_xpu_stitcher_build_parameters(ctx->ctx);
    if (chk != 0)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("构建相机参数失败, 内存不足或其它错误."));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL, END);
    }

    ctx->optimize_seam = optimize_seam;

    for (int i = 0; i < 10; i++)
    {
        ctx->src_file_p[i] = abcdk_option_get(ctx->args, "--src-video-file", i, NULL);
        if (!ctx->src_file_p[i])
            break;

        if (!ctx->src_file_p[i][0]) // ignore.
            continue;

        ctx->src_count += 1;
        ctx->src_pts[ctx->src_count - 1] = INT64_MIN;
        ctx->src_locker[ctx->src_count - 1] = abcdk_rwlock_create();
        ctx->src_dev_ctx[ctx->src_count - 1] = abcdk_xpu_context_refer(ctx->dev_ctx);
    }

    if (ctx->src_count < 2)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("源视频至少需要两路."));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL, END);
    }

    ctx->dst_dts = 0;
    ctx->dst_locker = abcdk_rwlock_create();
    ctx->dst_dev_ctx = abcdk_xpu_context_refer(ctx->dev_ctx);

    abcdk_worker_config_t cfg;
    cfg.numbers = ctx->src_count + 2;
    cfg.opaque = ctx;
    cfg.process_cb = _videostitcher_process_cb;

    ctx->worker_ctx = abcdk_worker_start(&cfg);

    abcdk_worker_dispatch(ctx->worker_ctx, 101, NULL);
    abcdk_worker_dispatch(ctx->worker_ctx, 102, NULL);

    for (int i = 0; i < ctx->src_count; i++)
    {
        abcdk_worker_dispatch(ctx->worker_ctx, i, NULL);
    }

    /*等待终止信号.*/
    abcdk_proc_wait_exit_signal(-1);

    /*通知所有作业退出.*/
    ctx->worker_exit_flag = 1;

END:

    abcdk_worker_stop(&ctx->worker_ctx);

    for (int i = 0; i < ctx->src_count; i++)
        abcdk_xpu_image_free(&ctx->src_img[i]);

    abcdk_xpu_image_free(&ctx->dst_img);

    abcdk_xpu_stitcher_free(&ctx->ctx);

    abcdk_xpu_context_current_set(NULL);
    abcdk_xpu_context_unref(&ctx->dev_ctx);

    abcdk_xpu_runtime_deinit();

    return;
}

int abcdk_tool_videostitcher(abcdk_option_t *args)
{
    videostitcher_t ctx = {0};

    ctx.args = args;

    if (abcdk_option_exist(ctx.args, "--help"))
    {
        _videostitcher_print_usage(ctx.args);
    }
    else
    {
        _videostitcher_work(&ctx);
    }

    return ctx.errcode;
}