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

    abcdk_rtsp_server_t *rtsp_ctx;


#define ABCDK_TOOL_VIDEOSTITCHER_SRC_MAX 10

    int src_count;
    const char *src_file_p[ABCDK_TOOL_VIDEOSTITCHER_SRC_MAX];
    const char *src_undistort_param_magic_p[ABCDK_TOOL_VIDEOSTITCHER_SRC_MAX];
    const char *src_undistort_param_file_p[ABCDK_TOOL_VIDEOSTITCHER_SRC_MAX];

    const char *dst_name_p;
    int dst_fps;

    int64_t src_pts[ABCDK_TOOL_VIDEOSTITCHER_SRC_MAX];
    abcdk_xpu_image_t *src_img[ABCDK_TOOL_VIDEOSTITCHER_SRC_MAX];
    abcdk_rwlock_t *src_locker[ABCDK_TOOL_VIDEOSTITCHER_SRC_MAX];

    int64_t dst_dts;
    abcdk_xpu_image_t *dst_img;
    abcdk_rwlock_t *dst_locker;

    abcdk_xpu_context_t *reader_dev_ctx[ABCDK_TOOL_VIDEOSTITCHER_SRC_MAX];
    abcdk_xpu_context_t *stitching_dev_ctx;
    abcdk_xpu_context_t *writer_dev_ctx;

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

    fprintf(stderr, "\n\t--rtsp-server-port < PORT >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t RTSP服务端口. 默认: 8554 \n"));
    
    fprintf(stderr, "\n\t--hwaccel-vendor < VENDOR > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 硬件加速供应商. 默认: %d\n"), ABCDK_XPU_HWACCEL_NONE);
    fprintf(stderr, ABCDK_GETTEXT("\t %d: 无\n"), ABCDK_XPU_HWACCEL_NONE);
    fprintf(stderr, ABCDK_GETTEXT("\t %d: 英伟达\n"), ABCDK_XPU_HWACCEL_NVIDIA);

    fprintf(stderr, "\n\t--device-id < ID > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 设备ID. 默认: 0\n"));

    fprintf(stderr, "\n\t--stitching-warper-name < NAME > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 拼接缝合矫正算法. 默认: spherical\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t plane: 平面\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t spherical: 球面\n"));

    fprintf(stderr, "\n\t--stitching-optimize-seam < BOOL > \n");
    fprintf(stderr, ABCDK_GETTEXT("\t 拼接缝合接缝美化. 默认: 1\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t 0: 关\n"));
    fprintf(stderr, ABCDK_GETTEXT("\t 1: 开\n"));

    fprintf(stderr, "\n\t--stitching-camera-param-magic < STRING >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 拼接缝合相机参数魔法字符串. 默认: ABCDK \n"));

    fprintf(stderr, "\n\t--stitching-camera-param-file < FILE >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 拼接缝合相机参数文件. 未指定则忽略.\n"));

    fprintf(stderr, "\n\t--dst-video-name < STRING >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 全景视频名称. 默认: output \n"));

    fprintf(stderr, "\n\t--dst-video-fps < NUMBER >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 全景视频帧率. 默认: 16 \n"));

    fprintf(stderr, "\n\t--src-video-file < FILE >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 源视频文件(包括路径). \n"));

    fprintf(stderr, "\n\t--src-undistort-param-magic < STRING >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 源视频畸变矫正参数魔法字符串. 默认: ABCDK \n"));

    fprintf(stderr, "\n\t--src-undistort-param-file < FILE >\n");
    fprintf(stderr, ABCDK_GETTEXT("\t 源视频畸变矫正参数文件.\n"));
}

void _videostitcher_reader(videostitcher_t *ctx, int id)
{
    abcdk_xpu_calibrate_t *undistort_ctx = NULL;
    abcdk_xpu_vdec_t *dec_ctx = NULL;
    abcdk_ffmpeg_editor_t *ff_ctx = NULL;
    abcdk_ffmpeg_editor_param_t ff_param = {0};
    AVPacket *ff_pkt = av_packet_alloc();
    int ff_stream_idx = -1;
    int ff_stream_eof = 0;
    abcdk_xpu_image_t *src_img = NULL;
    abcdk_xpu_image_t *src_img2 = NULL;
    int64_t src_pts = INT64_MIN;
    int chk,chk2;

    abcdk_xpu_context_current_set(ctx->reader_dev_ctx[id]);

    if (ctx->src_undistort_param_file_p[id] && access(ctx->src_undistort_param_file_p[id],R_OK) == 0)
    {
        undistort_ctx = abcdk_xpu_calibrate_alloc();
        assert(undistort_ctx != NULL);

        chk = abcdk_xpu_calibrate_load_parameters_from_file(undistort_ctx, ctx->src_undistort_param_file_p[id], ctx->src_undistort_param_magic_p[id]);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("READER[%d]加载矫正参数文件失败, 不存在或无权限."), id);
            goto END;
        }

        chk = abcdk_xpu_calibrate_build_parameters(undistort_ctx, 0);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("READER[%d]构建矫正参数失败, 内存不足或其它."),id);
            goto END;
        }
    }

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
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("READER[%d]到末尾或已断开, 2秒稍后重试."), id);
            sleep(2);
        }

        abcdk_xpu_vdec_free(&dec_ctx);
        abcdk_ffmpeg_editor_free(&ff_ctx);

        dec_ctx = abcdk_xpu_vdec_alloc();
        assert(dec_ctx != NULL);

        ff_ctx = abcdk_ffmpeg_editor_alloc(0);
        assert(ff_ctx != NULL);

        chk = abcdk_ffmpeg_editor_open(ff_ctx, &ff_param);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("READER[%d]打开失败, 不存在或无权限."), id);
            continue;
        }

        ff_stream_eof = 0;

        for (ff_stream_idx = 0; ff_stream_idx < abcdk_ffmpeg_editor_stream_nb(ff_ctx); ff_stream_idx++)
        {
            AVStream *p = abcdk_ffmpeg_editor_stream_ctx(ff_ctx, ff_stream_idx);

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

            /*先从解码器中取出图像, 以便释放解码空间.*/
            chk = abcdk_xpu_vdec_recv_frame(dec_ctx, &src_img, &src_pts);
            if (chk < 0)
            {
                abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("READER[%d]从解码器获取图像失败, 内存不足或其它."), id);
                break;
            }
            else if (chk > 0)
            {
                /*统一图像格式.*/
                chk = abcdk_xpu_imgproc_convert2(&src_img, ABCDK_XPU_PIXFMT_RGB24);
                if(chk != 0)
                {
                    abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("READER[%d]转换图像格式失败, 内存不足或其它."), id);
                    break;
                }

                if (undistort_ctx)
                {
                    chk = abcdk_xpu_calibrate_undistort(undistort_ctx, src_img, &src_img2, ABCDK_XPU_INTER_CUBIC);
                    if (chk != 0)
                    {
                        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("内存不足."));
                        break;
                    }
                }

                abcdk_rwlock_wrlock(ctx->src_locker[id], 1); // lock

                ctx->src_pts[id] = src_pts;                                                        // copy
                chk = abcdk_xpu_image_clone(src_img2 ? src_img2 : src_img, &ctx->src_img[id], 16); // clone

                abcdk_rwlock_unlock(ctx->src_locker[id]); // unlock.
                if (chk != 0)
                {
                    abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("READER[%d]复制图像失败, 内存不足或其它错误."), id);
                    break;
                }

                abcdk_trace_printf(LOG_DEBUG, "READER[%d],PTS[%.3f]", id, abcdk_ffmpeg_editor_stream_ts2sec(ff_ctx, ff_stream_idx, ctx->src_pts[id]));
                continue;
            }
            else if (chk == 0 && ff_stream_eof)
            {
                break;
            }

            chk = abcdk_ffmpeg_editor_read_packet(ff_ctx, ff_pkt);
            if (chk != 0)
            {
                ff_stream_eof = 1;
                abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("READER[%d]拉流失败, 到末尾或已断开."), id);
                continue;
            }

            /*向解码器发送数据包.*/
            chk = abcdk_xpu_vdec_send_packet(dec_ctx, ff_pkt->data, ff_pkt->size, ff_pkt->pts);
            if (chk < 0)
            {
                abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("READER[%d]向解码器发送数据失败, 内存不足或其它."), id);
                break;
            }
        }
    }

END:

    abcdk_xpu_image_free(&src_img);
    abcdk_xpu_image_free(&src_img2);

    abcdk_xpu_vdec_free(&dec_ctx);
    abcdk_ffmpeg_editor_free(&ff_ctx);
    av_packet_free(&ff_pkt);

    abcdk_xpu_calibrate_free(&undistort_ctx);

    abcdk_xpu_context_current_set(NULL);
    abcdk_xpu_context_unref(&ctx->reader_dev_ctx[id]);
}

void _videostitcher_stitching(videostitcher_t *ctx)
{
    int src_count;
    int64_t src_pts[10] = {INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN, INT64_MIN};
    abcdk_xpu_image_t *src_img[10] = {NULL};
    int64_t frame_duration = 0;
    int chk;

    abcdk_xpu_context_current_set(ctx->stitching_dev_ctx);

    frame_duration = (1000 / ctx->dst_fps);

    while (1)
    {
        if (abcdk_atomic_compare(&ctx->worker_exit_flag, 1))
            break;

        usleep(frame_duration * 1000); // nFPS.

        for (int i = 0; i < ctx->src_count; i++)
        {
            abcdk_rwlock_rdlock(ctx->src_locker[i], 1);//lock.

            if (ctx->src_img[i] && !abcdk_xpu_image_empty(ctx->src_img[i]))
            {
                src_pts[i] = ctx->src_pts[i];                            // copy
                abcdk_xpu_image_clone(ctx->src_img[i], &src_img[i], 16); // clone
            }

            abcdk_rwlock_unlock(ctx->src_locker[i]);//unlock.
        }

        src_count = 0; // set to 0.
        for (int i = 0; i < ctx->src_count; i++)
        {
            if (src_img[i] && !abcdk_xpu_image_empty(src_img[i]))
                src_count += 1;
        }

        if (src_count != ctx->src_count)
            continue;

        abcdk_rwlock_wrlock(ctx->dst_locker, 1);//lock.

        /*拼接.*/
        chk = abcdk_xpu_stitcher_compose(ctx->ctx, src_count, (const abcdk_xpu_image_t **)src_img, &ctx->dst_img, ctx->optimize_seam);
        /*滚动DTS.*/
        if (chk == 0)
            ctx->dst_dts += 1;

        abcdk_rwlock_unlock(ctx->dst_locker);//unlock.
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("全景拼接失败, 内存不足或其它错误."));
            break;
        }

        abcdk_trace_printf(LOG_DEBUG, "STITCHING,DTS[%lld]", ctx->dst_dts);
    }

    abcdk_xpu_context_current_set(NULL);
    abcdk_xpu_context_unref(&ctx->stitching_dev_ctx);
}

void _videostitcher_writer(videostitcher_t *ctx)
{
    abcdk_xpu_venc_t *enc_ctx = NULL;
    abcdk_xpu_vcodec_params_t enc_params = {0};
    abcdk_xpu_vcodec_params_t enc_params2 = {0};
    int rtsp_stream_id = -1;
    int64_t dst_dts = INT64_MIN;
    int64_t dst_dts2 = INT64_MIN;
    int64_t dst_pts = INT64_MIN;
    abcdk_xpu_image_t *dst_img = NULL;
    abcdk_xpu_image_t *dst_img2 = NULL;
    abcdk_object_t *dst_packet = NULL;
    int64_t frame_duration = 0;
    int chk;

    abcdk_xpu_context_current_set(ctx->writer_dev_ctx);

    enc_params.format = ABCDK_XPU_VCODEC_ID_H265;
    enc_params.bitrate = 15000 * 1000;     // 15Mbps
    enc_params.max_bitrate = 30000 * 1000; // 30Mbps
    enc_params.width = 1920;
    enc_params.height = 1080;
    enc_params.fps_n = ctx->dst_fps;
    enc_params.fps_d = 1;
    enc_params.max_b_frames = 0;
    enc_params.refs = 4;
    enc_params.hw_preset_type = 0;
    enc_params.idr_interval = 12;
    enc_params.iframe_interval = 13;
    enc_params.insert_spspps_idr = 50;
    enc_params.mode_vbr = 0;
    enc_params.level = 51;
    enc_params.profile = 66;
    enc_params.qmax = 51;
    enc_params.qmin = 25;

    enc_ctx = abcdk_xpu_venc_alloc();

    chk = abcdk_xpu_venc_setup(enc_ctx, &enc_params);
    assert(chk == 0);

    chk = abcdk_xpu_venc_get_params(enc_ctx, &enc_params2);
    assert(chk == 0);

    chk = abcdk_rtsp_server_create_media(ctx->rtsp_ctx, ctx->dst_name_p, "Video Stitcher", NULL);
    assert(chk == 0);

    abcdk_object_t *extdata = abcdk_object_copyfrom(enc_params2.ext_data, enc_params2.ext_size);
    rtsp_stream_id = abcdk_rtsp_server_add_stream(ctx->rtsp_ctx, ctx->dst_name_p, ABCDK_RTSP_CODEC_H265, extdata,
                                                  ABCDK_CLAMP(enc_params2.bitrate / 1000, 3000, 50000),
                                                  ABCDK_CLAMP(enc_params2.fps_n, 16, 100));
    abcdk_object_unref(&extdata);

    chk = abcdk_rtsp_server_play_media(ctx->rtsp_ctx, ctx->dst_name_p);
    assert(chk == 0);

    frame_duration = (1000 / enc_params2.fps_n);

    while (1)
    {
        if (abcdk_atomic_compare(&ctx->worker_exit_flag, 1))
            break;

        usleep(frame_duration * 1000); // nFPS.

        chk = abcdk_xpu_venc_recv_packet(enc_ctx, &dst_packet, &dst_pts);
        if (chk > 0)
        {
            AVRational timebase = {enc_params2.fps_d, enc_params2.fps_n};// 常用设定: 1/FPS

            double pts_sec = (double)(dst_pts)*abcdk_ffmpeg_q2d(&timebase, 1.); // 秒.
            double dur_sec = (double)(frame_duration);                          // 毫秒.

            chk = abcdk_rtsp_server_play_stream(ctx->rtsp_ctx, ctx->dst_name_p, rtsp_stream_id,
                                                dst_packet->pptrs[0], dst_packet->sizes[0],
                                                pts_sec * 1000000, dur_sec * 1000); // 转微秒.
            assert(chk == 0);

            abcdk_trace_printf(LOG_DEBUG, "WRITER,PTS[%.3f]", pts_sec);
        }

        abcdk_rwlock_rdlock(ctx->dst_locker, 1);//lock.

        if (ctx->dst_img && !abcdk_xpu_image_empty(ctx->dst_img))
        {
            dst_dts = ctx->dst_dts;                            // copy
            abcdk_xpu_image_clone(ctx->dst_img, &dst_img, 16); // clone
        }

        abcdk_rwlock_unlock(ctx->dst_locker);//unlock.

        /*序号没更新, 即表示全景图没有更新.*/
        if (dst_dts2 >= dst_dts)
            continue;

        /*更新序号.*/
        dst_dts2 = dst_dts;

        chk = abcdk_xpu_image_reset(&dst_img2, enc_params2.width, enc_params2.height, ABCDK_XPU_PIXFMT_RGB24, 16);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("内存不足."));
            goto END;
        }

        chk = abcdk_xpu_imgproc_resize(dst_img, NULL, dst_img2, ABCDK_XPU_INTER_CUBIC);
        if (chk != 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("内存不足."));
            goto END;
        }

        chk = abcdk_xpu_venc_send_frame(enc_ctx, dst_img2, dst_dts);
        if (chk < 0)
        {
            abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("内存不足."));
            goto END;
        }
    }

END:

    abcdk_object_unref(&dst_packet);

    abcdk_xpu_image_free(&dst_img);
    abcdk_xpu_image_free(&dst_img2);

    abcdk_xpu_venc_free(&enc_ctx);

    abcdk_rtsp_server_remove_media(ctx->rtsp_ctx,ctx->dst_name_p);

    abcdk_xpu_context_current_set(NULL);
    abcdk_xpu_context_unref(&ctx->writer_dev_ctx);
}

void _videostitcher_process_cb(void *opaque, uint64_t event, void *item)
{
    videostitcher_t *ctx = (videostitcher_t *)opaque;

    abcdk_thread_setname(pthread_self(), "vs-%llx", event);

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
    
    int rtsp_port = abcdk_option_get_int(ctx->args, "--rtsp-server-port", 0, 8554);
    int hwaccel_vendor = abcdk_option_get_int(ctx->args, "--hwaccel-vendor", 0, ABCDK_XPU_HWACCEL_NONE);
    int device_id = abcdk_option_get_int(ctx->args, "--device-id", 0, 0);
    ctx->optimize_seam = abcdk_option_get_int(ctx->args, "--stitching-optimize-seam", 0, 1);
    const char *warper_name_p = abcdk_option_get(ctx->args, "--stitching-warper-name", 0, "spherical");
    const char *camera_param_file_p = abcdk_option_get(ctx->args, "--stitching-camera-param-file", 0, NULL);
    const char *camera_param_magic_p = abcdk_option_get(ctx->args, "--stitching-camera-param-magic", 0, "ABCDK");
    ctx->dst_name_p = abcdk_option_get(ctx->args, "--dst-video-name", 0, "output");
    ctx->dst_fps = abcdk_option_get_int(ctx->args, "--dst-video-fps", 0, 16);

    ctx->rtsp_ctx = abcdk_rtsp_server_create(rtsp_port, 0x01 | 0x02 | 0x10);
    if (!ctx->rtsp_ctx)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("流媒体服务无法启动, 端口(%d)被占用或无权限."), rtsp_port);
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL, END);
    }

    chk = abcdk_rtsp_server_start(ctx->rtsp_ctx);
    assert(chk == 0);

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

    for (int i = 0; i < ABCDK_ARRAY_SIZE(ctx->src_file_p); i++)
    {
        ctx->src_file_p[i] = abcdk_option_get(ctx->args, "--src-video-file", i, NULL);
        ctx->src_undistort_param_magic_p[i] = abcdk_option_get(ctx->args, "--src-undistort-param-magic", i, NULL);
        ctx->src_undistort_param_file_p[i] = abcdk_option_get(ctx->args, "--src-undistort-param-file", i, "ABCDK");

        if (!ctx->src_file_p[i])
            break;

        ctx->src_count += 1;
        ctx->src_pts[ctx->src_count - 1] = INT64_MIN;
        ctx->src_locker[ctx->src_count - 1] = abcdk_rwlock_create();
    }

    if (ctx->src_count < 2)
    {
        abcdk_trace_printf(LOG_ERR, ABCDK_GETTEXT("源视频至少需要两路."));
        ABCDK_ERRNO_AND_GOTO1(ctx->errcode = -EINVAL, END);
    }

    ctx->dst_dts = 0;
    ctx->dst_locker = abcdk_rwlock_create();

    for (int i = 0; i < ctx->src_count; i++)
        ctx->reader_dev_ctx[i] = abcdk_xpu_context_refer(ctx->dev_ctx);

    ctx->stitching_dev_ctx = abcdk_xpu_context_refer(ctx->dev_ctx);
    ctx->writer_dev_ctx = abcdk_xpu_context_refer(ctx->dev_ctx);

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
    abcdk_atomic_store(&ctx->worker_exit_flag,1);

END:

    abcdk_worker_stop(&ctx->worker_ctx);

    for (int i = 0; i < ctx->src_count; i++)
        abcdk_xpu_image_free(&ctx->src_img[i]);

    abcdk_xpu_image_free(&ctx->dst_img);

    if(ctx->rtsp_ctx)
        abcdk_rtsp_server_stop(ctx->rtsp_ctx);
    abcdk_rtsp_server_destroy(&ctx->rtsp_ctx);

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