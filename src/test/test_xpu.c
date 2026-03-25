/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "entry.h"

static void _test_xpu_1(abcdk_option_t *args)
{
    int chk;

    for (int i = 0; i < ABCDK_XPU_PIXFMT_BGR32 + 1; i++)
    {
        int bit = abcdk_xpu_pixfmt_get_bit((abcdk_xpu_pixfmt_t)i, 1);
        const char *name = abcdk_xpu_pixfmt_get_name((abcdk_xpu_pixfmt_t)i);
        int channel = abcdk_xpu_pixfmt_get_channel((abcdk_xpu_pixfmt_t)i);

        if (bit <= 0)
            continue;

        abcdk_trace_printf(LOG_DEBUG, "bit:%d name:%s channel:%d", bit, name, channel);
    }

    abcdk_xpu_image_t *img_src = abcdk_xpu_image_alloc();
    abcdk_xpu_image_t *img_dst = abcdk_xpu_image_alloc();

    chk = abcdk_xpu_image_reset(&img_src, 100, 100, ABCDK_XPU_PIXFMT_BGR24, 16);
    assert(chk == 0);
    chk = abcdk_xpu_image_reset(&img_dst, 100, 100, ABCDK_XPU_PIXFMT_NV12, 16);
    assert(chk == 0);

    // chk = abcdk_xpu_image_copy(img_src,img_dst);
    // assert(chk == 0);

    chk = abcdk_xpu_imgproc_convert(img_src, img_dst);
    assert(chk == 0);

    abcdk_xpu_image_t *img_dst2 = abcdk_xpu_image_alloc();
    chk = abcdk_xpu_image_reset(&img_dst2, 1000, 1000, ABCDK_XPU_PIXFMT_NV12, 32);
    assert(chk == 0);

    chk = abcdk_xpu_imgproc_resize(img_src, NULL, img_dst2, ABCDK_XPU_INTER_CUBIC);
    assert(chk == 0);

    chk = abcdk_xpu_imgproc_resize(img_dst2, NULL, img_dst, ABCDK_XPU_INTER_CUBIC);
    assert(chk == 0);

    abcdk_xpu_image_free(&img_dst2);

    abcdk_xpu_image_free(&img_src);
    abcdk_xpu_image_free(&img_dst);
}

static void _test_xpu_2(abcdk_option_t *args)
{
    int chk;

    const char *src_file = abcdk_option_get(args, "--src-file", 0, "");
    const char *dst_file = abcdk_option_get(args, "--dst-file", 0, "");
    const char *dst_ext = abcdk_option_get(args, "--dst-ext", 0, ".jpg");

    abcdk_object_t *src_data = abcdk_object_copyfrom_file(src_file);

    abcdk_xpu_image_t *img_src = abcdk_xpu_imgcodec_decode(src_data->pptrs[0], src_data->sizes[0]);
    abcdk_object_unref(&src_data);

    // abcdk_object_t *dst_data2 = abcdk_xpu_imgcodec_encode(img_src, dst_ext);
    // abcdk_save(dst_file, dst_data2->pptrs[0], dst_data2->sizes[0], 0);
    // abcdk_object_unref(&dst_data2);

    abcdk_xpu_point_t p1 = {100, 100};
    abcdk_xpu_point_t p2 = {200, 200};
    abcdk_xpu_scalar_t scalar = {.u8[0] = 255, .u8[1] = 0, .u8[2] = 0};

    abcdk_xpu_imgproc_line(img_src, &p1, &p2, &scalar, 30);

    abcdk_xpu_rect_t rect = {333, 333, 444, 444};

    abcdk_xpu_imgproc_rectangle(img_src, &rect, 3, &scalar);

    abcdk_xpu_rect_t rect2 = {444, 444, 555, 555};

    abcdk_xpu_imgproc_stuff(img_src, &rect2, &scalar);

    abcdk_xpu_scalar_t alpha = {.f32[0] = 1, .f32[1] = 1, .f32[2] = 1};
    abcdk_xpu_scalar_t bate = {.f32[0] = 100, .f32[1] = 100, .f32[2] = 100};

    abcdk_xpu_imgproc_brightness(img_src, &alpha, &bate);

    abcdk_object_t *dst_data = abcdk_xpu_imgcodec_encode(img_src, dst_ext);
    abcdk_dump(dst_file, dst_data->pptrs[0], dst_data->sizes[0]);
    abcdk_object_unref(&dst_data);

    abcdk_xpu_image_free(&img_src);
}

static void _test_xpu_3(abcdk_option_t *args)
{
    int chk;

    const char *src_file = abcdk_option_get(args, "--src-file", 0, "");
    const char *dst_file = abcdk_option_get(args, "--dst-file", 0, "");

    abcdk_object_t *src_data = abcdk_object_copyfrom_file(src_file);

    abcdk_xpu_image_t *img_src = abcdk_xpu_imgcodec_decode(src_data->pptrs[0], src_data->sizes[0]);
    abcdk_object_unref(&src_data);

    abcdk_xpu_image_t *img_dst = abcdk_xpu_image_alloc();
    abcdk_xpu_image_reset(&img_dst, 1000, 2000, ABCDK_XPU_PIXFMT_BGR24, 32);

    abcdk_xpu_rect_t src_roi = {100,200, 500,500};

 //   chk = abcdk_xpu_imgproc_resize(img_src, &src_roi, img_dst, ABCDK_XPU_INTER_CUBIC);
    chk = abcdk_xpu_imgproc_resize(img_src, NULL, img_dst, ABCDK_XPU_INTER_CUBIC);
    assert(chk == 0);

    abcdk_object_t *dst_data = abcdk_xpu_imgcodec_encode(img_dst, ".jpg");
    abcdk_dump(dst_file, dst_data->pptrs[0], dst_data->sizes[0]);
    abcdk_object_unref(&dst_data);

    abcdk_xpu_image_free(&img_dst);
    abcdk_xpu_image_free(&img_src);
}

static int _load_imgs(abcdk_xpu_image_t *img[100], const char *img_path)
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

        img[count] = abcdk_xpu_imgcodec_decode_from_file(file);
        assert(img[count]);

        char tmp_file2[100] = {0};
        snprintf(tmp_file2, 100, "/tmp/ddd/load_imgs-%d.jpg", count);

        abcdk_xpu_imgcodec_encode_to_file(img[count], tmp_file2, ".jpg");

        count += 1;
    }

    abcdk_tree_free(&dir_ctx);

    return count;
}

static void _test_xpu_4(abcdk_option_t *args)
{
    int chk;

    const char *dst_file = abcdk_option_get(args, "--dst-file", 0, "");
    const char *src_file = abcdk_option_get(args, "--src-file", 0, "");
    const char *param_file = abcdk_option_get(args, "--calibrate-param", 0, "calibrate.xml");
    const char *magic_name = abcdk_option_get(args, "--magic-name", 0, "abcdk");

    abcdk_xpu_image_t *src_imgs[100] = {0};
    _load_imgs(src_imgs, src_file);

    abcdk_xpu_calibrate_t *ctx = abcdk_xpu_calibrate_alloc();

    chk = abcdk_xpu_calibrate_load_parameters_from_file(ctx, param_file, magic_name);
    if (chk != 0)
    {
        abcdk_xpu_calibrate_setup(ctx, 7, 9, 40, 40);

        for (int i = 0; i < 100; i++)
        {
            if (!src_imgs[i])
                break;

            abcdk_xpu_calibrate_detect_corners(ctx, src_imgs[i],5,5);
        }

        double rms = abcdk_xpu_calibrate_estimate_parameters(ctx);

        abcdk_trace_printf(LOG_DEBUG, "RMS: %.3lf", rms);

        chk = abcdk_xpu_calibrate_dump_parameters_to_file(ctx, param_file, magic_name);
        assert(chk == 0);
    }

    chk = abcdk_xpu_calibrate_build_parameters(ctx, 1);
    assert(chk == 0);

    abcdk_xpu_image_t *dst_img = NULL;

    chk = abcdk_xpu_calibrate_undistort(ctx, src_imgs[0], &dst_img, ABCDK_XPU_INTER_CUBIC);
    assert(chk == 0);

    chk = abcdk_xpu_imgcodec_encode_to_file(dst_img, dst_file, NULL);
    assert(chk == 0);

    abcdk_xpu_image_free(&dst_img);

    for (int i = 0; i < 100; i++)
        abcdk_xpu_image_free(&src_imgs[i]);

    abcdk_xpu_calibrate_free(&ctx);
}

static void _test_xpu_5(abcdk_option_t *args)
{
    int chk;

    const char *dst_file = abcdk_option_get(args, "--dst-file", 0, "");
    const char *src_file = abcdk_option_get(args, "--src-file", 0, "");
    const char *feature_name = abcdk_option_get(args, "--feature-name", 0, "SURF");
    const char *warper_name = abcdk_option_get(args, "--warper-name", 0, "spherical");
    int optimize_seam = abcdk_option_get_int(args, "--optimize-seam", 0, 1);
    const char *param_file = abcdk_option_get(args, "--stitcher-param", 0, "stitcher.xml");
    const char *magic_name = abcdk_option_get(args, "--magic-name", 0, "abcdk");

    abcdk_xpu_image_t *src_imgs[100] = {0};
    int count = _load_imgs(src_imgs, src_file);

    abcdk_xpu_stitcher_t *ctx = abcdk_xpu_stitcher_alloc();

    chk = abcdk_xpu_stitcher_set_feature_finder(ctx, feature_name);
    assert(chk == 0);

    chk = abcdk_xpu_stitcher_set_warper(ctx, warper_name);
    assert(chk == 0);

    chk = abcdk_xpu_stitcher_load_parameters_from_file(ctx, param_file, magic_name);
    if (chk != 0)
    {
        chk = abcdk_xpu_stitcher_estimate_parameters(ctx, count, (const abcdk_xpu_image_t **)src_imgs, NULL, 0.8);
        assert(chk == 0);

        chk = abcdk_xpu_stitcher_build_parameters(ctx);
        assert(chk == 0);

        chk = abcdk_xpu_stitcher_dump_parameters_to_file(ctx, param_file, magic_name);
        assert(chk == 0);
    }
    else
    {
        chk = abcdk_xpu_stitcher_build_parameters(ctx);
        assert(chk == 0);
    }

    abcdk_xpu_image_t *dst_img = NULL;

    chk = abcdk_xpu_stitcher_compose(ctx, count, (const abcdk_xpu_image_t **)src_imgs, &dst_img, optimize_seam);
    assert(chk == 0);

    abcdk_xpu_stitcher_free(&ctx);

    chk = abcdk_xpu_imgcodec_encode_to_file(dst_img, dst_file, NULL);
    assert(chk == 0);

    abcdk_xpu_image_free(&dst_img);

    for (int i = 0; i < 100; i++)
        abcdk_xpu_image_free(&src_imgs[i]);
}

static void _test_xpu_6(abcdk_option_t *args)
{
    int chk;

    const char *dst_file = abcdk_option_get(args, "--dst-file", 0, "");
    const char *src_file = abcdk_option_get(args, "--src-file", 0, "");

    abcdk_xpu_venc_t *venc_ctx = abcdk_xpu_venc_alloc();

    abcdk_xpu_vcodec_params_t venc_params = {0};

    // venc_params.format = ABCDK_XPU_VCODEC_ID_H264;
    venc_params.format = ABCDK_XPU_VCODEC_ID_H265;
    venc_params.bitrate = 15000 * 1000;     // 15Mbps
    venc_params.max_bitrate = 30000 * 1000; // 30Mbps
    venc_params.width = 1920;
    venc_params.height = 1080;
    venc_params.fps_n = 25;
    venc_params.fps_d = 1;
    venc_params.max_b_frames = 0;
    venc_params.refs = 4;
    venc_params.hw_preset_type = 0;
    venc_params.idr_interval = 12;
    venc_params.iframe_interval = 13;
    venc_params.insert_spspps_idr = 50;
    venc_params.mode_vbr = 0;
    venc_params.level = 51;
    venc_params.profile = 66;
    venc_params.qmax = 51;
    venc_params.qmin = 25;

    chk = abcdk_xpu_venc_setup(venc_ctx, &venc_params);
    assert(chk == 0);

    abcdk_xpu_vcodec_params_t venc_params2 = {0};
    chk = abcdk_xpu_venc_get_params(venc_ctx, &venc_params2);
    assert(chk == 0);

    int dst_fd = abcdk_open(dst_file, 1, 0, 1);

    abcdk_write(dst_fd, venc_params2.ext_data, venc_params2.ext_size);

    abcdk_xpu_image_t *src = abcdk_xpu_imgcodec_decode_from_file(src_file);

    for (int i = 0; i < 100; i++)
    {
        abcdk_object_t *dst = NULL;
        int64_t dts;
        chk = abcdk_xpu_venc_recv_packet(venc_ctx, &dst, &dts);
        if (chk == 1)
        {
            abcdk_trace_printf(LOG_DEBUG, "dts:%lld", dts);
            abcdk_write(dst_fd, dst->pptrs[0], dst->sizes[0]);
        }
        abcdk_object_unref(&dst);

        chk = abcdk_xpu_venc_send_frame(venc_ctx, src, i - 1);
        assert(chk >= 0);
    }

    chk = abcdk_xpu_venc_send_frame(venc_ctx, NULL, 0);
    assert(chk >= 0);

    for (int i = 0; i < 100; i++)
    {
        abcdk_object_t *dst = NULL;
        int64_t dts;
        chk = abcdk_xpu_venc_recv_packet(venc_ctx, &dst, &dts);
        if (chk != 1)
            break;

        abcdk_trace_printf(LOG_DEBUG, "dts:%lld", dts);

        abcdk_write(dst_fd, dst->pptrs[0], dst->sizes[0]);
        abcdk_object_unref(&dst);
    }

    abcdk_xpu_image_free(&src);

    abcdk_closep(&dst_fd);

    abcdk_xpu_venc_free(&venc_ctx);
}

static void _test_xpu_7(abcdk_option_t *args)
{
#ifdef HAVE_FFMPEG

    int chk;

    const char *dst_file = abcdk_option_get(args, "--dst-file", 0, "");
    const char *src_file = abcdk_option_get(args, "--src-file", 0, "");

    abcdk_xpu_vdec_t *vdec_ctx = abcdk_xpu_vdec_alloc();

    abcdk_ffmpeg_editor_t *ff_ctx = abcdk_ffmpeg_editor_alloc(0);

    abcdk_ffmpeg_editor_param_t ff_param = {0};

    ff_param.url = src_file;
    ff_param.timeout = 0;
    ff_param.read_mp4toannexb = 1;
    ff_param.read_ignore_audio = 1;
    ff_param.read_ignore_subtitle = 1;
    ff_param.read_nodelay = 1;

    chk = abcdk_ffmpeg_editor_open(ff_ctx, &ff_param);
    assert(chk == 0);

    for (int i = 0; i < abcdk_ffmpeg_editor_stream_nb(ff_ctx); i++)
    {
        AVStream *p = abcdk_ffmpeg_editor_stream_ctx(ff_ctx, i);

        if (p->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            abcdk_xpu_vcodec_params_t venc_params = {0};

            if (p->codecpar->codec_id == AV_CODEC_ID_H264)
                venc_params.format = ABCDK_XPU_VCODEC_ID_H264;
            if (p->codecpar->codec_id == AV_CODEC_ID_H265)
                venc_params.format = ABCDK_XPU_VCODEC_ID_H265;

            venc_params.ext_data = p->codecpar->extradata;
            venc_params.ext_size = p->codecpar->extradata_size;

            chk = abcdk_xpu_vdec_setup(vdec_ctx, &venc_params);
            assert(chk == 0);
        }
    }

    AVPacket *ff_pkt = av_packet_alloc();
    abcdk_xpu_image_t *dst_img = NULL;
    int64_t dst_ts = 0;

    for (int i = 0; i < 100000; i++)
    {
        chk = abcdk_ffmpeg_editor_read_packet(ff_ctx, ff_pkt);
        if (chk != 0)
            break;

        chk = abcdk_xpu_vdec_recv_frame(vdec_ctx, &dst_img, &dst_ts);
        if (chk > 0)
        {
            abcdk_trace_printf(LOG_DEBUG, "pts:%.3f", abcdk_ffmpeg_editor_stream_ts2sec(ff_ctx, ff_pkt->stream_index, dst_ts));
            
            static int save_ok = 0;
            if (!save_ok)
            {
                chk = abcdk_xpu_imgcodec_encode_to_file(dst_img, dst_file, NULL);
                save_ok = (chk == 0 ? 1 : 0);
            }
        }

        chk = abcdk_xpu_vdec_send_packet(vdec_ctx, ff_pkt->data, ff_pkt->size, ff_pkt->pts);
        assert(chk > 0);
    }

    chk = abcdk_xpu_vdec_send_packet(vdec_ctx, NULL, 0, 0);
    assert(chk > 0);

    for (;;)
    {
        chk = abcdk_xpu_vdec_recv_frame(vdec_ctx, &dst_img, &dst_ts);
        if (chk <= 0)
            break;

        abcdk_trace_printf(LOG_DEBUG, "pts:%.3f", abcdk_ffmpeg_editor_stream_ts2sec(ff_ctx, ff_pkt->stream_index, dst_ts));
        chk = abcdk_xpu_imgcodec_encode_to_file(dst_img, dst_file, NULL);
    }

    abcdk_xpu_image_free(&dst_img);

    av_packet_free(&ff_pkt);

    abcdk_ffmpeg_editor_free(&ff_ctx);

    abcdk_xpu_vdec_free(&vdec_ctx);

#endif //HAVE_FFMPEG
}

static uint8_t select_color(int idx, int channel)
{
    assert(idx >= 0 && channel >= 0);

    static int tables[][3] = {
        {0, 114, 189},
        {217, 83, 25},
        {237, 177, 32},
        {126, 47, 142},
        {119, 172, 48},
        {77, 190, 238},
        {162, 20, 47},
        {76, 76, 76},
        {153, 153, 153},
        {255, 0, 0},
        {255, 128, 0},
        {191, 191, 0},
        {0, 255, 0},
        {0, 0, 255},
        {170, 0, 255},
        {85, 85, 0},
        {85, 170, 0},
        {85, 255, 0},
        {170, 85, 0},
        {170, 170, 0},
        {170, 255, 0},
        {255, 85, 0},
        {255, 170, 0},
        {255, 255, 0},
        {0, 85, 128},
        {0, 170, 128},
        {0, 255, 128},
        {85, 0, 128},
        {85, 85, 128},
        {85, 170, 128},
        {85, 255, 128},
        {170, 0, 128},
        {170, 85, 128},
        {170, 170, 128},
        {170, 255, 128},
        {255, 0, 128},
        {255, 85, 128},
        {255, 170, 128},
        {255, 255, 128},
        {0, 85, 255},
        {0, 170, 255},
        {0, 255, 255},
        {85, 0, 255},
        {85, 85, 255},
        {85, 170, 255},
        {85, 255, 255},
        {170, 0, 255},
        {170, 85, 255},
        {170, 170, 255},
        {170, 255, 255},
        {255, 0, 255},
        {255, 85, 255},
        {255, 170, 255},
        {85, 0, 0},
        {128, 0, 0},
        {170, 0, 0},
        {212, 0, 0},
        {255, 0, 0},
        {0, 43, 0},
        {0, 85, 0},
        {0, 128, 0},
        {0, 170, 0},
        {0, 212, 0},
        {0, 255, 0},
        {0, 0, 43},
        {0, 0, 85},
        {0, 0, 128},
        {0, 0, 170},
        {0, 0, 212},
        {0, 0, 255},
        {0, 0, 0},
        {36, 36, 36},
        {73, 73, 73},
        {109, 109, 109},
        {146, 146, 146},
        {182, 182, 182},
        {219, 219, 219},
        {0, 114, 189},
        {80, 183, 189},
        {128, 128, 0},
        {255, 56, 56},
        {255, 157, 151},
        {255, 112, 31},
        {255, 178, 29},
        {207, 210, 49},
        {72, 249, 10},
        {146, 204, 23},
        {61, 219, 134},
        {26, 147, 52},
        {0, 212, 187},
        {44, 153, 168},
        {0, 194, 255},
        {52, 69, 147},
        {100, 115, 255},
        {0, 24, 236},
        {132, 56, 255},
        {82, 0, 133},
        {203, 56, 255},
        {255, 149, 200},
        {255, 55, 199}};

    return tables[idx % 100][channel % 3];
}

static void idx2nyxz(size_t idx, size_t h, size_t w, size_t c)
{
    size_t n = idx / (h * w * c);
    size_t y = (idx / (w * c)) % h;
    size_t x = (idx / c) % w;
    size_t z = idx % c;

    abcdk_trace_printf(LOG_DEBUG, "[%zu][%zu][%zu][%zu]", n, y, x, z);
}

static void _size_dst2src(abcdk_xpu_dnn_object_t *obj, double src_w, double src_h, double dst_w, double dst_h, int keep_ratio)
{
    abcdk_resize_scale_t r = {0};

    abcdk_resize_ratio_2d(&r, src_w, src_h, dst_w, dst_h, keep_ratio);

    for (int k = 0; k < obj->rect.nb; k++)
    {
        abcdk_xpu_point_t *pt_p = &obj->rect.pt[k];

        pt_p->x = abcdk_resize_dst2src_2d(&r, pt_p->x, 1);
        pt_p->y = abcdk_resize_dst2src_2d(&r, pt_p->y, 0);
    }

    for (int k = 0; k < obj->rrect.nb; k++)
    {
        abcdk_xpu_point_t *pt_p = &obj->rrect.pt[k];

        pt_p->x = abcdk_resize_dst2src_2d(&r, pt_p->x, 1);
        pt_p->y = abcdk_resize_dst2src_2d(&r, pt_p->y, 0);
    }

    for (int k = 0; k < obj->nkeypoint * 3; k += 3)
    {
        obj->kp[k + 0] = abcdk_resize_dst2src_2d(&r, obj->kp[k + 0], 1);
        obj->kp[k + 1] = abcdk_resize_dst2src_2d(&r, obj->kp[k + 1], 0);
    }
}

static int _test_xpu_8(abcdk_option_t *args)
{

    // for(int i = 0;i<1000;i++)
    //     idx2nyxz(i,3,3,3);

    int test_count = abcdk_option_get_int(args, "--test-count", 0, 1);

    const char *model_p = abcdk_option_get(args, "--model", 0, "");
    const char *model_name_p = abcdk_option_get(args, "--model-name", 0, "yolo-v11");
    const char *img_src = abcdk_option_get(args, "--img-src", 0, "");

    int chk;

    abcdk_xpu_dnn_post_t *post_ctx = abcdk_xpu_dnn_post_alloc();

    abcdk_xpu_dnn_post_init(post_ctx, model_name_p, args);

    abcdk_xpu_dnn_infer_t *infer_ctx = abcdk_xpu_dnn_infer_alloc();

    chk = abcdk_xpu_dnn_infer_load_model(infer_ctx, model_p, args);
    assert(chk == 0);

    abcdk_xpu_dnn_tensor_t vec_tensor[100];

    int tensor_num = abcdk_xpu_dnn_infer_fetch_tensor(infer_ctx, 100, vec_tensor);
    assert(tensor_num >= 2);

    abcdk_xpu_image_t *vec_img[100] = {0};
    int count = 0;

    const char *img_src_p = abcdk_option_get(args, "--img-src", 0, "");
    count = _load_imgs(vec_img,img_src_p);

    for (int t = 0; t < test_count; t++)
    {
        uint64_t s = abcdk_time_systime(9);

        abcdk_clock(s, &s);

        chk = abcdk_xpu_dnn_infer_forward(infer_ctx, count, vec_img);
        assert(chk == 0);

        uint64_t step = abcdk_clock(s, &s);

        abcdk_trace_printf(LOG_INFO, "step: %.6lf", ((double)step) / 1000000000.);
    }

    abcdk_xpu_dnn_post_process(post_ctx, tensor_num, vec_tensor, 0.1, 0.1);

    abcdk_xpu_dnn_tensor_t *input_tensor_p = &vec_tensor[0];

    for (int i = 0; i < input_tensor_p->dims.d[0]; i++)
    {
        abcdk_xpu_dnn_object_t vec_obj[100] = {0};
        chk = abcdk_xpu_dnn_post_fetch(post_ctx, i, 100, vec_obj);
        if (chk <= 0)
            continue;

        abcdk_xpu_image_t *img_p = vec_img[i];

        for (int j = 0; j < chk; j++)
        {
            abcdk_xpu_dnn_object_t *obj_p = &vec_obj[j];

            _size_dst2src(obj_p, abcdk_xpu_image_get_width(img_p), abcdk_xpu_image_get_height(img_p), input_tensor_p->dims.d[3], input_tensor_p->dims.d[2], 0);

            abcdk_xpu_scalar_t color = {255, 0, 0};
            int weight = 3;
            int corner[4] = {obj_p->rect.pt[0].x, obj_p->rect.pt[0].y, obj_p->rect.pt[1].x, obj_p->rect.pt[1].y};

            //   abcdk_xpu_imgproc_drawrect(img_p, color, weight, corner);

            //   abcdk_trace_printf(LOG_INFO, "r=%d", obj_p->angle);

            int idx = rand();

            color.u8[0] = select_color(idx, 0);
            color.u8[1] = select_color(idx, 1);
            color.u8[2] = select_color(idx, 2);

            for (int k = 0; k < obj_p->rrect.nb; k++)
            {
                abcdk_xpu_point_t *pt1_p = &obj_p->rrect.pt[k];
                abcdk_xpu_point_t *pt2_p = &obj_p->rrect.pt[(k + 1) % obj_p->rrect.nb];
#if 1

                abcdk_xpu_imgproc_line(img_p, pt1_p, pt2_p, &color, weight);
#else
                corner[0] = pt_p->x - 3;
                corner[1] = pt_p->y - 3;
                corner[2] = pt_p->x + 3;
                corner[3] = pt_p->y + 3;

                abcdk_xpu_imgproc_drawrect(img_p, color, weight, corner);
#endif
            }

            for (int k = 0; k < obj_p->nkeypoint * 3; k += 3)
            {
                abcdk_xpu_rect_t rect;

                rect.x = obj_p->kp[k + 0] - 10;
                rect.y = obj_p->kp[k + 1] - 10;
                rect.width = 20;
                rect.height = 20;

                weight = 5;

                abcdk_xpu_imgproc_rectangle(img_p, &rect, weight, &color);
            }

            if (obj_p->seg)
            {
                abcdk_xpu_image_t *seg_img_src = abcdk_xpu_image_create(input_tensor_p->dims.d[3], input_tensor_p->dims.d[2], ABCDK_XPU_PIXFMT_GRAYF32, 4);
                abcdk_xpu_image_t *seg_img_dst = abcdk_xpu_image_create(abcdk_xpu_image_get_width(img_p), abcdk_xpu_image_get_height(img_p), ABCDK_XPU_PIXFMT_GRAYF32, 4);

                uint8_t *seg_data[4] = {(uint8_t *)obj_p->seg, 0, 0, 0};
                int seg_linesize[4] = {obj_p->seg_step, -1, -1, -1};

                abcdk_xpu_image_upload((const uint8_t **)seg_data, (const int *)seg_linesize, seg_img_src);

                abcdk_xpu_imgproc_resize(seg_img_src,NULL, seg_img_dst,  ABCDK_XPU_INTER_CUBIC);

                abcdk_xpu_imgproc_mask(img_p, seg_img_dst, 0.5, &color, 1);

                abcdk_xpu_image_free(&seg_img_src);
                abcdk_xpu_image_free(&seg_img_dst);
            }
        }

        char tmp_file[100] = {0};
        snprintf(tmp_file, 100, "/tmp/ccc/dnn-%d.jpg", i);

        abcdk_xpu_imgcodec_encode_to_file(img_p, tmp_file, ".jpg");
    }

    abcdk_xpu_dnn_post_free(&post_ctx);

    for (int i = 0; i < 100; i++)
        abcdk_xpu_image_free(&vec_img[i]);

    abcdk_xpu_dnn_infer_free(&infer_ctx);

    return 0;
}

static int _test_xpu_9(abcdk_option_t *args)
{
    const char *src_p = abcdk_option_get(args, "--src-file", 0, "");
    const char *dst_p = abcdk_option_get(args, "--dst-file", 0, "");

    abcdk_xpu_image_t *src_img = abcdk_xpu_imgcodec_decode_from_file(src_p);
    abcdk_xpu_image_t *dst_img = NULL;

#if 1

    abcdk_xpu_point_t dst_quad[4] = {
        {30, 30},   // 左上角
        {220, 50},  // 右上角
        {210, 220}, // 右下角
        {50, 230},  // 左下角
    };

    abcdk_xpu_point_t src_quad[4] = {
        {86, 136},  // 左上角
        {173, 186}, // 右上角
        {123, 273}, // 右下角
        {36, 223},  // 左下角
    };
    
#else

    abcdk_xpu_point_t dst_quad[4] = {760,350,1160,350,1220,440,820,440};

    abcdk_xpu_point_t src_quad[4] = {770,520,1150,520,1100,610,720,610};

#endif

    int weight = 3;
    abcdk_xpu_scalar_t src_color = {.u8[0] = 255};
    abcdk_xpu_scalar_t dst_color = {.u8[2] = 255};

    for (int i = 0; i < 4; i++)
        abcdk_xpu_imgproc_line(src_img, &src_quad[i], &src_quad[(i + 1) % 4], &src_color, weight);

    dst_img = abcdk_xpu_imgcodec_decode_from_file(dst_p);

    for (int i = 0; i < 4; i++)
        abcdk_xpu_imgproc_line(dst_img, &dst_quad[i], &dst_quad[(i + 1) % 4], &dst_color, weight);

    abcdk_xpu_imgproc_warp_quad2quad(src_img, src_quad, dst_img, dst_quad, 1, ABCDK_XPU_INTER_LINEAR);
    abcdk_xpu_imgcodec_encode_to_file(dst_img, "/tmp/test.warp-1.jpg", NULL);
    abcdk_xpu_image_free(&dst_img);

    dst_img = abcdk_xpu_imgcodec_decode_from_file(dst_p);

    for (int i = 0; i < 4; i++)
        abcdk_xpu_imgproc_line(dst_img, &dst_quad[i], &dst_quad[(i + 1) % 4], &dst_color, weight);
    abcdk_xpu_imgproc_warp_quad2quad(src_img, src_quad, dst_img, dst_quad, 2, ABCDK_XPU_INTER_LINEAR);
    abcdk_xpu_imgcodec_encode_to_file(dst_img, "/tmp/test.warp-2.jpg", NULL);
    abcdk_xpu_image_free(&dst_img);

    dst_img = abcdk_xpu_imgcodec_decode_from_file(dst_p);

    for (int i = 0; i < 4; i++)
        abcdk_xpu_imgproc_line(dst_img, &dst_quad[i], &dst_quad[(i + 1) % 4], &dst_color, weight);
    abcdk_xpu_imgproc_warp_quad2quad(src_img, NULL, dst_img, dst_quad, 1, ABCDK_XPU_INTER_LINEAR);
    abcdk_xpu_imgcodec_encode_to_file(dst_img, "/tmp/test-nosrc.warp-1.jpg", NULL);
    abcdk_xpu_image_free(&dst_img);

    dst_img = abcdk_xpu_imgcodec_decode_from_file(dst_p);

    for (int i = 0; i < 4; i++)
        abcdk_xpu_imgproc_line(dst_img, &dst_quad[i], &dst_quad[(i + 1) % 4], &dst_color, weight);

    abcdk_xpu_imgproc_warp_quad2quad(src_img, NULL, dst_img, dst_quad, 2, ABCDK_XPU_INTER_LINEAR);
    abcdk_xpu_imgcodec_encode_to_file(dst_img, "/tmp/test-nosrc.warp-2.jpg", NULL);
    abcdk_xpu_image_free(&dst_img);

    dst_img = abcdk_xpu_imgcodec_decode_from_file(dst_p);

    for (int i = 0; i < 4; i++)
        abcdk_xpu_imgproc_line(dst_img, &dst_quad[i], &dst_quad[(i + 1) % 4], &dst_color, weight);

    abcdk_xpu_imgproc_warp_quad2quad(src_img, src_quad, dst_img, NULL, 1, ABCDK_XPU_INTER_LINEAR);
    abcdk_xpu_imgcodec_encode_to_file(dst_img, "/tmp/test-nodst.warp-1.jpg", NULL);
    abcdk_xpu_image_free(&dst_img);

    dst_img = abcdk_xpu_imgcodec_decode_from_file(dst_p);

    for (int i = 0; i < 4; i++)
        abcdk_xpu_imgproc_line(dst_img, &dst_quad[i], &dst_quad[(i + 1) % 4], &dst_color, weight);

    abcdk_xpu_imgproc_warp_quad2quad(src_img, src_quad, dst_img, NULL, 2, ABCDK_XPU_INTER_LINEAR);
    abcdk_xpu_imgcodec_encode_to_file(dst_img, "/tmp/test-nodst.warp-2.jpg", NULL);
    abcdk_xpu_image_free(&dst_img);

    abcdk_xpu_image_free(&src_img);
}

static int _test_xpu_10(abcdk_option_t *args)
{
    const char *src_p = abcdk_option_get(args, "--src-file", 0, "");
    const char *dst_p = abcdk_option_get(args, "--dst-file", 0, "");

    abcdk_xpu_image_t *src_img = abcdk_xpu_imgcodec_decode_from_file(src_p);

    abcdk_xpu_image_t *dst_img = abcdk_xpu_image_create(200, 90, ABCDK_XPU_PIXFMT_BGR24, 16);

    abcdk_xpu_point_t src_quad[4] = {
        {257, 173},  // 左上角
        {370, 173}, // 右上角
        {377, 227}, // 右下角
        {264, 230},  // 左下角
    };

    abcdk_xpu_imgproc_quad2rect(src_img, src_quad, dst_img, ABCDK_XPU_INTER_CUBIC);

    abcdk_xpu_imgcodec_encode_to_file(dst_img, dst_p, NULL);

    abcdk_xpu_image_free(&dst_img);

    abcdk_xpu_image_free(&src_img);
}

static int _test_xpu_11(abcdk_option_t *args)
{

    int test_count = abcdk_option_get_int(args, "--test-count", 0, 1);

    const char *model_p = abcdk_option_get(args, "--model", 0, "");
    const char *model_name_p = abcdk_option_get(args, "--model-name", 0, "face-yunet");
    const char *img_src = abcdk_option_get(args, "--img-src", 0, "");

    int chk;

    abcdk_xpu_dnn_post_t *post_ctx = abcdk_xpu_dnn_post_alloc();

    abcdk_xpu_dnn_post_init(post_ctx, model_name_p, args);

    abcdk_xpu_dnn_infer_t *infer_ctx = abcdk_xpu_dnn_infer_alloc();

    chk = abcdk_xpu_dnn_infer_load_model(infer_ctx, model_p, args);
    assert(chk == 0);

    abcdk_xpu_dnn_tensor_t vec_tensor[100];

    int tensor_num = abcdk_xpu_dnn_infer_fetch_tensor(infer_ctx, 100, vec_tensor);
    assert(tensor_num >= 2);

    abcdk_xpu_image_t *vec_img[100] = {0};
    int count = 0;

    const char *img_src_p = abcdk_option_get(args, "--img-src", 0, "");
    count = _load_imgs(vec_img,img_src_p);

    chk = abcdk_xpu_dnn_infer_forward(infer_ctx, count, vec_img);
    assert(chk == 0);

    abcdk_xpu_dnn_post_process(post_ctx, tensor_num, vec_tensor, 0.1, 0.1);

    abcdk_xpu_dnn_tensor_t *input_tensor_p = &vec_tensor[0];

    abcdk_xpu_image_t *dst_img = NULL;

    for (int i = 0; i < input_tensor_p->dims.d[0]; i++)
    {
        abcdk_xpu_dnn_object_t vec_obj[100] = {0};
        chk = abcdk_xpu_dnn_post_fetch(post_ctx, i, 100, vec_obj);
        if (chk <= 0)
            continue;

        abcdk_xpu_image_t *img_p = vec_img[i];

        for (int j = 0; j < chk; j++)
        {
            abcdk_xpu_dnn_object_t *obj_p = &vec_obj[j];

            _size_dst2src(obj_p, abcdk_xpu_image_get_width(img_p), abcdk_xpu_image_get_height(img_p), input_tensor_p->dims.d[3], input_tensor_p->dims.d[2], 0);

            abcdk_xpu_point_t face_kpt[5];
            for (int k = 0; k < obj_p->nkeypoint * 3; k += 3)
            {
                face_kpt[k / 3].x = obj_p->kp[k + 0];
                face_kpt[k / 3].y = obj_p->kp[k + 1];
            }

            abcdk_xpu_imgproc_face_warp(img_p, face_kpt, &dst_img, ABCDK_XPU_INTER_CUBIC);

            char tmp_file[100] = {0};
            snprintf(tmp_file, 100, "/tmp/ccc/face-%d.jpg", j);

            abcdk_xpu_imgcodec_encode_to_file(dst_img, tmp_file, ".jpg");
        }
    }

    abcdk_xpu_image_free(&dst_img);

    abcdk_xpu_dnn_post_free(&post_ctx);

    for (int i = 0; i < 100; i++)
        abcdk_xpu_image_free(&vec_img[i]);

    abcdk_xpu_dnn_infer_free(&infer_ctx);

    return 0;
}

static int _test_xpu_12(abcdk_option_t *args)
{

    int test_count = abcdk_option_get_int(args, "--test-count", 0, 1);

    const char *model_p = abcdk_option_get(args, "--model", 0, "");
    const char *model_name_p = abcdk_option_get(args, "--model-name", 0, "face-sface");
    const char *img_src = abcdk_option_get(args, "--img-src", 0, "");

    int chk;

    abcdk_xpu_dnn_post_t *post_ctx = abcdk_xpu_dnn_post_alloc();

    abcdk_xpu_dnn_post_init(post_ctx, model_name_p, args);

    abcdk_xpu_dnn_infer_t *infer_ctx = abcdk_xpu_dnn_infer_alloc();

    chk = abcdk_xpu_dnn_infer_load_model(infer_ctx, model_p, args);
    assert(chk == 0);

    abcdk_xpu_dnn_tensor_t vec_tensor[100];

    int tensor_num = abcdk_xpu_dnn_infer_fetch_tensor(infer_ctx, 100, vec_tensor);
    assert(tensor_num >= 2);

    abcdk_xpu_image_t *vec_img[100] = {0};
    int count = 0;

    const char *img_src_p = abcdk_option_get(args, "--img-src", 0, "");
    count = _load_imgs(vec_img,img_src_p);

    chk = abcdk_xpu_dnn_infer_forward(infer_ctx, count, vec_img);
    assert(chk == 0);

    abcdk_xpu_dnn_post_process(post_ctx, tensor_num, vec_tensor, 0.1, 0.1);

    abcdk_xpu_dnn_tensor_t *input_tensor_p = &vec_tensor[0];

    abcdk_xpu_image_t *dst_img = NULL;

    for (int i = 0; i < input_tensor_p->dims.d[0]; i++)
    {
        abcdk_xpu_dnn_object_t vec_obj[100] = {0};
        chk = abcdk_xpu_dnn_post_fetch(post_ctx, i, 100, vec_obj);
        if (chk <= 0)
            continue;

        abcdk_xpu_image_t *img_p = vec_img[i];

        for (int j = 0; j < chk; j++)
        {
            abcdk_xpu_dnn_object_t *obj_p = &vec_obj[j];

           
        }
    }

    abcdk_xpu_image_free(&dst_img);

    abcdk_xpu_dnn_post_free(&post_ctx);

    for (int i = 0; i < 100; i++)
        abcdk_xpu_image_free(&vec_img[i]);

    abcdk_xpu_dnn_infer_free(&infer_ctx);

    return 0;
}

static int _test_xpu_13(abcdk_option_t *args)
{
#ifdef HAVE_FFMPEG
    const char *dst_file_p = NULL;

    int src_file_max = 4;
    int src_file_num = 0;
    const char *src_file_p[4] = {NULL};

    dst_file_p = abcdk_option_get(args, "--dst-file", 0, "");

    for (int i = 0; i < src_file_max; i++)
    {
        src_file_p[i] = abcdk_option_get(args, "--src-file", i, "");
        if (!src_file_p[i])
            break;

        src_file_num += 1;
    }

    abcdk_xpu_vdec_t *src_dec_ctx[4] = {NULL};
    abcdk_ffmpeg_editor_t *src_ff_ctx[4] = {NULL};
    int chk;

    for (int i = 0; i < src_file_max; i++)
    {
        src_dec_ctx[i] = abcdk_xpu_vdec_alloc();

        src_ff_ctx[i] = abcdk_ffmpeg_editor_alloc(0);

        abcdk_ffmpeg_editor_param_t ff_param = {0};

        ff_param.url = src_file_p[i];
        ff_param.timeout = 0;
        ff_param.read_mp4toannexb = 1;
        ff_param.read_ignore_audio = 1;
        ff_param.read_ignore_subtitle = 1;
        ff_param.read_nodelay = 1;

        chk = abcdk_ffmpeg_editor_open(src_ff_ctx[i], &ff_param);
        assert(chk == 0);

        for (int j = 0; j < abcdk_ffmpeg_editor_stream_nb(src_ff_ctx[i]); j++)
        {
            AVStream *p = abcdk_ffmpeg_editor_stream_ctx(src_ff_ctx[i], j);

            if (p->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                abcdk_xpu_vcodec_params_t src_dec_params = {0};

                if (p->codecpar->codec_id == AV_CODEC_ID_H264)
                    src_dec_params.format = ABCDK_XPU_VCODEC_ID_H264;
                if (p->codecpar->codec_id == AV_CODEC_ID_H265)
                    src_dec_params.format = ABCDK_XPU_VCODEC_ID_H265;

                src_dec_params.ext_data = p->codecpar->extradata;
                src_dec_params.ext_size = p->codecpar->extradata_size;

                chk = abcdk_xpu_vdec_setup(src_dec_ctx[j], &src_dec_params);
                assert(chk == 0);
            }
        }
    }

    abcdk_xpu_venc_t *dst_enc_ctx = abcdk_xpu_venc_alloc();

    abcdk_xpu_vcodec_params_t dst_enc_params = {0};

    // dst_enc_params.format = ABCDK_XPU_VCODEC_ID_H264;
    dst_enc_params.format = ABCDK_XPU_VCODEC_ID_H265;
    dst_enc_params.bitrate = 15000 * 1000;     // 15Mbps
    dst_enc_params.max_bitrate = 30000 * 1000; // 30Mbps
    dst_enc_params.width = 1920;
    dst_enc_params.height = 1080;
    dst_enc_params.fps_n = 25;
    dst_enc_params.fps_d = 1;
    dst_enc_params.max_b_frames = 0;
    dst_enc_params.refs = 4;
    dst_enc_params.hw_preset_type = 0;
    dst_enc_params.idr_interval = 12;
    dst_enc_params.iframe_interval = 13;
    dst_enc_params.insert_spspps_idr = 50;
    dst_enc_params.mode_vbr = 0;
    dst_enc_params.level = 51;
    dst_enc_params.profile = 66;
    dst_enc_params.qmax = 51;
    dst_enc_params.qmin = 25;

    chk = abcdk_xpu_venc_setup(dst_enc_ctx, &dst_enc_params);
    assert(chk == 0);

    abcdk_xpu_vcodec_params_t dst_enc_params2 = {0};
    chk = abcdk_xpu_venc_get_params(dst_enc_ctx, &dst_enc_params2);
    assert(chk == 0);

    abcdk_ffmpeg_editor_t *dst_ff_ctx = NULL;

    dst_ff_ctx = abcdk_ffmpeg_editor_alloc(1);

    abcdk_ffmpeg_editor_param_t dst_ff_param = {0};

    dst_ff_param.url = dst_file_p;
    dst_ff_param.timeout = 0;
    dst_ff_param.write_nodelay = 1;
    dst_ff_param.write_fmp4 = 1;

    chk = abcdk_ffmpeg_editor_open(dst_ff_ctx, &dst_ff_param);
    assert(chk == 0);

    // AVRational time_base = {25, 1};
    // chk = abcdk_ffmpeg_editor_add_stream2(dst_ff_ctx, &dst_enc_params2, &time_base, NULL, NULL);
    // assert(chk >= 0);

    abcdk_xpu_image_t *src_img_ori[4] = {0};
    abcdk_xpu_image_t *src_img_fix[4] = {0};
    int src_recv_count = 0;

    for (int i = 0; i < src_file_max; i++)
    {
        AVPacket *src_ff_pkt = av_packet_alloc();
        int src_recv_ok = 0;

        for (int j = 0; j < 100; j++)
        {
            if (src_recv_ok)
                break;

            chk = abcdk_ffmpeg_editor_read_packet(src_ff_ctx[i], src_ff_pkt);
            if (chk != 0)
                break;

            int64_t dst_ts;

            chk = abcdk_xpu_vdec_recv_frame(src_dec_ctx[i], &src_img_ori[i], &dst_ts);
            if (chk > 0)
            {
                abcdk_trace_printf(LOG_DEBUG, "src[%d],pts:%.3f", i + 1, abcdk_ffmpeg_editor_stream_ts2sec(src_ff_ctx[i], src_ff_pkt->stream_index, dst_ts));
                src_recv_ok = 1;
            }

            chk = abcdk_xpu_vdec_send_packet(src_dec_ctx[i], src_ff_pkt->data, src_ff_pkt->size, src_ff_pkt->pts);
            assert(chk > 0);
        }

        av_packet_free(&src_ff_pkt);

        if (src_recv_ok)
            src_recv_count += 1;

        if (src_recv_count == src_file_num)
            break;
    }
#endif //#ifdef HAVE_FFMPEG
}

int abcdk_test_xpu(abcdk_option_t *args)
{
    int hwaccel_vendor = abcdk_option_get_int(args, "--hwaccel-vendor", 0, ABCDK_XPU_HWACCEL_NONE);
    int device_id = abcdk_option_get_int(args, "--device-id", 0, 0);
    int cmd = abcdk_option_get_int(args, "--cmd", 0, 1);

    int chk = abcdk_xpu_runtime_init(hwaccel_vendor);
    assert(chk == 0);

    abcdk_xpu_context_t *ctx = abcdk_xpu_context_alloc(device_id);
    assert(ctx != NULL);

    abcdk_xpu_context_current_set(ctx);
    abcdk_xpu_context_current_set(ctx);
    abcdk_xpu_context_t *cp_ctx = abcdk_xpu_context_refer(ctx);

    if (cmd == 1)
        _test_xpu_1(args);
    if (cmd == 2)
        _test_xpu_2(args);
    if (cmd == 3)
        _test_xpu_3(args);
    if (cmd == 4)
        _test_xpu_4(args);
    if (cmd == 5)
        _test_xpu_5(args);
    if (cmd == 6)
        _test_xpu_6(args);
    if (cmd == 7)
        _test_xpu_7(args);
    if (cmd == 8)
        _test_xpu_8(args);
    if (cmd == 9)
        _test_xpu_9(args);
    if (cmd == 10)
        _test_xpu_10(args);
    if (cmd == 11)
        _test_xpu_11(args);
    if (cmd == 12)
        _test_xpu_12(args);
    if (cmd == 13)
        _test_xpu_13(args);

    abcdk_xpu_context_current_set(NULL);
    abcdk_xpu_context_unref(&ctx);
    abcdk_xpu_context_unref(&cp_ctx);

    abcdk_xpu_runtime_deinit();

    return 0;
}
