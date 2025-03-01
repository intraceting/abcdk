/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"

#ifdef  HAVE_CUDA 

int abcdk_test_cuda_1(abcdk_option_t *args, CUcontext cuda_ctx)
{
    int chk;

    abcdk_media_image_t *a = abcdk_cuda_image_create(200, 200, ABCDK_MEDIA_PIXFMT_RGB24, 123);
    abcdk_media_image_t *b = abcdk_cuda_image_create(200, 200, ABCDK_MEDIA_PIXFMT_RGB24, 234);

    uint8_t scalar[3] = {128, 255, 0};
    abcdk_cuda_imgproc_stuff_8u(3,1,a->data[0], a->width, a->stride[0], a->height, scalar);

    uint8_t color[3] = {255, 0, 0};
    int corner[4] = {10, 10, 100, 100};

    abcdk_cuda_imgproc_drawrect_8u(3,1,a->data[0], a->width, a->stride[0], a->height, color, 3, corner);

    chk = abcdk_cuda_image_copy(b, a);
    assert(chk == 0);

    // abcdk_bmp_save_file("/tmp/test.cuda.a.bmp",a->data[0],a->stride[0],a->width,a->height,24);
    abcdk_cuda_image_save("/tmp/test.cuda.b.bmp", b);

    //abcdk_media_image_t *c = abcdk_cuda_image_create(200, 200, ABCDK_MEDIA_PIXFMT_YUV420P, 567);
    abcdk_media_image_t *c = abcdk_cuda_image_create(200, 200, ABCDK_MEDIA_PIXFMT_NV21, 567);

    abcdk_cuda_image_convert(c, a);

    abcdk_media_image_t *d = abcdk_cuda_image_create(200, 200, ABCDK_MEDIA_PIXFMT_RGB24, 678);

    abcdk_cuda_image_convert(d, c);

    abcdk_cuda_image_save("/tmp/test.cuda.d.bmp", d);

    abcdk_media_image_t *e = abcdk_cuda_image_create(800, 600, ABCDK_MEDIA_PIXFMT_RGB24, 678);

    abcdk_cuda_imgproc_resize_8u(3,1,e->data[0],e->width,e->stride[0],e->height, NULL, d->data[0],d->width,d->stride[0],d->height, NULL, 1, NPPI_INTER_CUBIC);

    abcdk_cuda_image_save("/tmp/test.cuda.e.bmp", e);

    abcdk_media_image_t *f = abcdk_cuda_image_create(800, 600, ABCDK_MEDIA_PIXFMT_RGB24, 678);

    NppiPoint dst_quad[4] = {
        {30, 30},   // 变换后的左上角
        {220, 50},  // 变换后的右上角
        {210, 220}, // 变换后的右下角
        {50, 230},  // 变换后的左下角
    };

    NppiRect src_roi = {100, 100, 200, 200};

    abcdk_cuda_imgproc_warp_8u(3, 1, f->data[0], f->width, f->stride[0], f->height, NULL, dst_quad, e->data[0], e->width, e->stride[0], e->height, NULL, NULL, 1, NPPI_INTER_CUBIC);

    abcdk_cuda_image_save("/tmp/test.cuda.f.bmp", f);

    abcdk_cuda_jpeg_save("/tmp/test.cuda.f.jpg", f, cuda_ctx);
    // abcdk_cuda_jpeg_save("/tmp/test.cuda.f2.jpg", f);

    // abcdk_cuda_image_save("/tmp/test.cuda.f2.bmp", f);

    abcdk_media_image_free(&a);
    abcdk_media_image_free(&b);
    abcdk_media_image_free(&c);
    abcdk_media_image_free(&d);
    abcdk_media_image_free(&e);
    abcdk_media_image_free(&f);

    for (int i = 0; i < 10; i++)
    {
        abcdk_media_image_t *g = abcdk_cuda_jpeg_load("/tmp/test.cuda.f.jpg", cuda_ctx);

        abcdk_cuda_imgproc_drawrect_8u(3,1,g->data[0], g->width, g->stride[0], g->height, color, 3, corner);

        abcdk_cuda_image_save("/tmp/test.cuda.g2.bmp", g);
        abcdk_cuda_jpeg_save("/tmp/test.cuda.g2.jpg", g, cuda_ctx);

        abcdk_media_image_free(&g);
    }

    return 0;
}

#if 0

int abcdk_test_cuda_2(abcdk_option_t *args, CUcontext cuda_ctx)
{
    abcdk_ffeditor_config_t ff_r_cfg = {0};

    ff_r_cfg.file_name = abcdk_option_get(args, "--src", 0, "");
    ff_r_cfg.read_flush = abcdk_option_get_double(args, "--src-flush", 0, 0);
    ff_r_cfg.read_speed = abcdk_option_get_double(args, "--src-xpeed", 0, 1);
    ff_r_cfg.read_delay_max = abcdk_option_get_double(args, "--src-delay-max", 0, 10);
    ff_r_cfg.bit_stream_filter = 1;

    abcdk_ffeditor_t *r = abcdk_ffeditor_open(&ff_r_cfg);

    AVStream *r_video_steam = abcdk_ffeditor_find_stream(r, AVMEDIA_TYPE_VIDEO);

    abcdk_cuda_video_t *dec_ctx = abcdk_cuda_video_create(0, NULL, cuda_ctx);

    AVCodecContext *dec_opt = abcdk_avcodec_alloc3(r_video_steam->codecpar->codec_id, 0);
    abcdk_avstream_parameters_to_context(dec_opt, r_video_steam);
    abcdk_cuda_video_sync(dec_ctx, dec_opt);
    abcdk_avcodec_free(&dec_opt);

    AVPacket r_pkt;
    av_init_packet(&r_pkt);

    abcdk_cuda_jpeg_t *jpeg_w = abcdk_cuda_jpeg_create(1, NULL, cuda_ctx);

    for (int i = 0; i < 10000; i++)
    {
        int chk = abcdk_ffeditor_read_packet(r, &r_pkt, r_video_steam->index);
        if (chk < 0)
            break;

        AVFrame *r_fae = NULL;
        chk = abcdk_cuda_video_decode(dec_ctx, &r_fae, &r_pkt);
        if (chk < 0)
        {
            break;
        }
        else if (chk > 0)
        {
            char filename[PATH_MAX] = {0};
            sprintf(filename, "/tmp/ccc/%06d.jpg", r_fae->pts);

            abcdk_mkdir(filename, 0755);

            abcdk_cuda_jpeg_encode_to_file(jpeg_w, filename, r_fae);
        }

        av_frame_free(&r_fae);
    }

    abcdk_cuda_jpeg_destroy(&jpeg_w);

    av_packet_unref(&r_pkt);

    abcdk_cuda_video_destroy(&dec_ctx);
    abcdk_ffeditor_destroy(&r);

    return 0;
}

int abcdk_test_cuda_3(abcdk_option_t *args, CUcontext cuda_ctx)
{
    abcdk_ffeditor_config_t ff_r_cfg = {0}, ff_w_cfg = {1};

    ff_r_cfg.file_name = abcdk_option_get(args, "--src", 0, "");
    ff_r_cfg.read_flush = abcdk_option_get_double(args, "--src-flush", 0, 0);
    ff_r_cfg.read_speed = abcdk_option_get_double(args, "--src-xpeed", 0, 1);
    ff_r_cfg.read_delay_max = abcdk_option_get_double(args, "--src-delay-max", 0, 10);
    ff_r_cfg.bit_stream_filter = 1;
    ff_w_cfg.file_name = abcdk_option_get(args, "--dst", 0, "");
    ff_w_cfg.short_name = abcdk_option_get(args, "--dst-fmt", 0, "");

    abcdk_ffeditor_t *r = abcdk_ffeditor_open(&ff_r_cfg);
    abcdk_ffeditor_t *w = abcdk_ffeditor_open(&ff_w_cfg);

    AVStream *r_video_steam = abcdk_ffeditor_find_stream(r, AVMEDIA_TYPE_VIDEO);

    abcdk_cuda_video_t *dec_ctx = abcdk_cuda_video_create(0, NULL, cuda_ctx);
    abcdk_cuda_video_t *enc_ctx = abcdk_cuda_video_create(1, NULL, cuda_ctx);

    AVCodecContext *dec_opt = abcdk_avcodec_alloc3(r_video_steam->codecpar->codec_id, 0);
    abcdk_avstream_parameters_to_context(dec_opt, r_video_steam);
    abcdk_cuda_video_sync(dec_ctx, dec_opt);
    abcdk_avcodec_free(&dec_opt);

    AVCodecContext *enc_opt = abcdk_avcodec_alloc3(AV_CODEC_ID_HEVC, 0);

    abcdk_avcodec_encode_video_fill_time_base(enc_opt, 25);

    enc_opt->width = r_video_steam->codecpar->width;
    enc_opt->height = r_video_steam->codecpar->height;
    enc_opt->extradata = NULL;
    enc_opt->extradata_size = 0;
    enc_opt->max_b_frames = 0;
    enc_opt->bit_rate_tolerance = 15000 * 1000;
    enc_opt->bit_rate = 15000 * 1000;

    abcdk_cuda_video_sync(enc_ctx, enc_opt);

    int w_stream_idx = abcdk_ffeditor_add_stream(w, enc_opt, 1);

    abcdk_avcodec_free(&dec_opt);

    abcdk_ffeditor_write_header(w, 0);

    AVPacket r_pkt, *w_pkt = NULL;
    av_init_packet(&r_pkt);

    for (int i = 0; i < 10000; i++)
    {
        int chk = abcdk_ffeditor_read_packet(r, &r_pkt, r_video_steam->index);
        if (chk < 0)
            break;

        AVFrame *r_fae = NULL;
        chk = abcdk_cuda_video_decode(dec_ctx, &r_fae, &r_pkt);
        if (chk < 0)
        {
            break;
        }
        else if (chk > 0)
        {
            AVPacket *w_pkt = NULL;
            int chk = abcdk_cuda_video_encode(enc_ctx, &w_pkt, r_fae);
            if (chk <= 0)
                break;

            abcdk_ffeditor_write_packet2(w, w_pkt->data, w_pkt->size, w_pkt->flags, w_stream_idx);
            av_packet_free(&w_pkt);
        }

        av_frame_free(&r_fae);
    }

    av_packet_unref(&r_pkt);

    for (int i = 0; i < 1000; i++)
    {
        AVFrame *r_fae = NULL;
        AVPacket *w_pkt = NULL;

        if (i == 0)
        {
            /*通知解码器是结束包。*/
            int chk = abcdk_cuda_video_decode(dec_ctx, &r_fae, &r_pkt);
            if (chk < 0)
                break;
            else if (chk > 0)
            {
                chk = abcdk_cuda_video_encode(enc_ctx, &w_pkt, r_fae);
                if (chk <= 0)
                    break;

                abcdk_ffeditor_write_packet2(w, w_pkt->data, w_pkt->size, w_pkt->flags, w_stream_idx);
                av_packet_free(&w_pkt);
            }
        }
        else
        {
            int chk = abcdk_cuda_video_encode(enc_ctx, &w_pkt, NULL);
            if (chk <= 0)
                break;

            abcdk_ffeditor_write_packet2(w, w_pkt->data, w_pkt->size, w_pkt->flags, w_stream_idx);
            av_packet_free(&w_pkt);
        }
    }

    abcdk_ffeditor_write_trailer(w);

    abcdk_cuda_video_destroy(&dec_ctx);
    abcdk_cuda_video_destroy(&enc_ctx);
    abcdk_ffeditor_destroy(&r);
    abcdk_ffeditor_destroy(&w);

    return 0;
}

#endif //HAVE_FFMPEG

int abcdk_test_cuda_4(abcdk_option_t *args, CUcontext cuda_ctx)
{
    int n = 1, w = 300, h = 300 , depth =3;

    cuCtxPushCurrent(cuda_ctx);


    cuCtxPopCurrent(NULL);
}


int abcdk_test_cuda(abcdk_option_t *args)
{
    int cmd = abcdk_option_get_int(args, "--cmd", 0, 1);

    cuInit(0);

    int gpu = abcdk_option_get_int(args, "--gpu", 0, 0);

    int chk = abcdk_cuda_set_device(gpu);
    assert(chk == 0);

    char name[256] = {0};
    chk = abcdk_cuda_get_device_name(name, gpu);
    assert(chk == 0);

    fprintf(stderr, "%s\n", name);

    CUcontext cuda_ctx = abcdk_cuda_ctx_create(gpu, 0);

    if (cmd == 1)
        return abcdk_test_cuda_1(args, cuda_ctx);
#if 0
    else if (cmd == 2)
        return abcdk_test_cuda_2(args, cuda_ctx);
    else if (cmd == 3)
        return abcdk_test_cuda_3(args, cuda_ctx);
#endif //HAVE_FFMPEG
    else if (cmd == 4)
        return abcdk_test_cuda_4(args, cuda_ctx);


    abcdk_cuda_ctx_destroy(&cuda_ctx);

    return 0;
}

#else  // HAVE_CUDA && HAVE_FFMPEG

int abcdk_test_cuda(abcdk_option_t *args)
{
    return 0;
}

#endif // HAVE_CUDA && HAVE_FFMPEG
