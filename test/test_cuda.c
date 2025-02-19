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

// environ

#ifdef HAVE_CUDA
#ifdef HAVE_FFMPEG

int abcdk_test_cuda_1(abcdk_option_t *args)
{
    int gpu = abcdk_option_get_int(args, "--gpu", 0, 0);

    int chk = abcdk_cuda_set_device(gpu);
    assert(chk == 0);

    char name[256] = {0};
    chk = abcdk_cuda_get_device_name(name, gpu);
    assert(chk == 0);

    fprintf(stderr, "%s\n", name);

    // AVFrame *a = abcdk_cuda_avframe_alloc(100,1000,AV_PIX_FMT_YUV420P,1);
    // AVFrame *b = abcdk_avframe_alloc(100,1000,AV_PIX_FMT_YUV420P,1);
    // AVFrame *a = abcdk_cuda_avframe_alloc(100,1000,AV_PIX_FMT_NV16,123);
    // AVFrame *b = abcdk_avframe_alloc(100,1000,AV_PIX_FMT_NV16,234);
    // AVFrame *a = abcdk_cuda_avframe_alloc(100,1000,AV_PIX_FMT_NV12,123);
    // AVFrame *b = abcdk_avframe_alloc(100,1000,AV_PIX_FMT_NV12,234);
    AVFrame *a = abcdk_cuda_avframe_alloc(200, 200, AV_PIX_FMT_RGB24, 123);
    AVFrame *b = abcdk_avframe_alloc(200, 200, AV_PIX_FMT_RGB24, 234);

    uint8_t scalar[3] = {128, 255, 0};
    abcdk_cuda_imgproc_stuff_8u_c3r(a->data[0], a->width, a->linesize[0], a->height, scalar);

    uint8_t color[3] = {255, 0, 0};
    int corner[4] = {10, 10, 100, 100};

    abcdk_cuda_imgproc_drawrect_8u_c3r(a->data[0], a->width, a->linesize[0], a->height, color, 3, corner);

    chk = abcdk_cuda_avframe_copy(b, a);
    assert(chk == 0);

    // abcdk_bmp_save_file("/tmp/test.cuda.a.bmp",a->data[0],a->linesize[0],a->width,a->height,24);
    abcdk_cuda_avframe_save("/tmp/test.cuda.b.bmp", b);

    AVFrame *c = abcdk_cuda_avframe_alloc(200, 200, AV_PIX_FMT_YUV420P, 567);

    abcdk_cuda_avframe_convert(c, a);

    AVFrame *d = abcdk_cuda_avframe_alloc(200, 200, AV_PIX_FMT_RGB24, 678);

    abcdk_cuda_avframe_convert(d, c);

    AVFrame *e = abcdk_cuda_avframe_alloc(800, 600, AV_PIX_FMT_RGB24, 678);

    abcdk_cuda_avframe_resize(e, NULL, d, NULL, 1, NPPI_INTER_CUBIC);

    abcdk_cuda_avframe_save("/tmp/test.cuda.e.bmp", e);

    AVFrame *f = abcdk_cuda_avframe_alloc(800, 600, AV_PIX_FMT_RGB24, 678);

    NppiPoint dst_quad[4] = {
        {30, 30},   // 变换后的左上角
        {220, 50},  // 变换后的右上角
        {210, 220}, // 变换后的右下角
        {50, 230},  // 变换后的左下角
    };

    NppiRect src_roi = {100, 100, 200, 200};

    // abcdk_cuda_avframe_warp(f, NULL, dst_quad , e, &src_roi , NULL,2 , NPPI_INTER_CUBIC);
    abcdk_cuda_avframe_warp(f, NULL, dst_quad, e, NULL, NULL, 1, NPPI_INTER_CUBIC);

    abcdk_cuda_avframe_save("/tmp/test.cuda.f.bmp", f);

    abcdk_cuda_jpeg_save("/tmp/test.cuda.f.jpg", f);

    for (int i = 0; i < 1000; i++)
    {
        AVFrame *g = abcdk_cuda_jpeg_load("/tmp/test.cuda.f.jpg");

        abcdk_cuda_jpeg_save("/tmp/test.cuda.g.jpg", g);

        av_frame_free(&g);
    }

    av_frame_free(&a);
    av_frame_free(&b);
    av_frame_free(&c);
    av_frame_free(&d);
    av_frame_free(&e);
    av_frame_free(&f);

    return 0;
}

int abcdk_test_cuda_2(abcdk_option_t *args)
{
    abcdk_ffmpeg_config_t ff_r_cfg = {0};

    ff_r_cfg.file_name = abcdk_option_get(args,"--src",0,"");
    ff_r_cfg.read_flush = abcdk_option_get_double(args,"--src-flush",0,0);
    ff_r_cfg.read_speed = abcdk_option_get_double(args,"--src-xpeed",0,1);
    ff_r_cfg.read_delay_max = abcdk_option_get_double(args,"--src-delay-max",0,10);
    ff_r_cfg.bit_stream_filter = 1;

    abcdk_ffmpeg_t *r = abcdk_ffmpeg_open(&ff_r_cfg);

    AVStream *r_video_steam = abcdk_ffmpeg_find_stream(r,AVMEDIA_TYPE_VIDEO);

    abcdk_cuda_video_t *dec_ctx = abcdk_cuda_video_create(0,NULL);

    AVCodecContext *dec_opt = abcdk_avcodec_alloc3(r_video_steam->codecpar->codec_id,0);
    abcdk_avstream_parameters_to_context(dec_opt,r_video_steam);
    abcdk_cuda_video_sync(dec_ctx,dec_opt);
    abcdk_avcodec_free(&dec_opt);

    AVPacket r_pkt;
    av_init_packet(&r_pkt);

    abcdk_cuda_jpeg_t *jpeg_w = abcdk_cuda_jpeg_create(1,NULL);

    for (int i = 0; i < 10000; i++)
    {
        int chk = abcdk_ffmpeg_read_packet(r, &r_pkt, r_video_steam->index);
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

            abcdk_cuda_jpeg_encode_to_file(jpeg_w,filename, r_fae);
        }

        av_frame_free(&r_fae);
    }

    abcdk_cuda_jpeg_destroy(&jpeg_w);

    av_packet_unref(&r_pkt);

    abcdk_cuda_video_destroy(&dec_ctx);
    abcdk_ffmpeg_destroy(&r);

    return 0;
}


int abcdk_test_cuda_3(abcdk_option_t *args)
{
    abcdk_ffmpeg_config_t ff_r_cfg = {0},ff_w_cfg = {1};

    ff_r_cfg.file_name = abcdk_option_get(args,"--src",0,"");
    ff_r_cfg.read_flush = abcdk_option_get_double(args,"--src-flush",0,0);
    ff_r_cfg.read_speed = abcdk_option_get_double(args,"--src-xpeed",0,1);
    ff_r_cfg.read_delay_max = abcdk_option_get_double(args,"--src-delay-max",0,10);
    ff_r_cfg.bit_stream_filter = 1;
    ff_w_cfg.file_name = abcdk_option_get(args,"--dst",0,"");
    ff_w_cfg.short_name = abcdk_option_get(args,"--dst-fmt",0,"");

    abcdk_ffmpeg_t *r = abcdk_ffmpeg_open(&ff_r_cfg);
    abcdk_ffmpeg_t *w = abcdk_ffmpeg_open(&ff_w_cfg);

    AVStream *r_video_steam = abcdk_ffmpeg_find_stream(r,AVMEDIA_TYPE_VIDEO);

    abcdk_cuda_video_t *dec_ctx = abcdk_cuda_video_create(0,NULL);
    abcdk_cuda_video_t *enc_ctx = abcdk_cuda_video_create(1,NULL);

    AVCodecContext *dec_opt = abcdk_avcodec_alloc3(r_video_steam->codecpar->codec_id,0);
    abcdk_avstream_parameters_to_context(dec_opt,r_video_steam);
    abcdk_cuda_video_sync(dec_ctx,dec_opt);
    abcdk_avcodec_free(&dec_opt);

    AVCodecContext *enc_opt = abcdk_avcodec_alloc3(AV_CODEC_ID_H264,1);

    abcdk_avcodec_encode_video_fill_time_base(enc_opt, 25);

    enc_opt->width = r_video_steam->codecpar->width;
    enc_opt->height = r_video_steam->codecpar->height;
    enc_opt->extradata = NULL;
    enc_opt->extradata_size = 0;
    enc_opt->max_b_frames = 0;
    enc_opt->bit_rate_tolerance = 15000 * 1000;
    enc_opt->bit_rate = 15000 * 1000;

    abcdk_cuda_video_sync(enc_ctx,enc_opt);

    int w_stream_idx = abcdk_ffmpeg_add_stream(w, enc_opt, 1);

    abcdk_avcodec_free(&dec_opt);

    abcdk_ffmpeg_write_header(w,0);

    AVPacket r_pkt,*w_pkt=NULL;
    av_init_packet(&r_pkt);

    for (int i = 0; i < 10000; i++)
    {
        int chk = abcdk_ffmpeg_read_packet(r, &r_pkt, r_video_steam->index);
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

            abcdk_ffmpeg_write_packet2(w, w_pkt->data, w_pkt->size, w_pkt->flags, w_stream_idx);
            av_packet_free(&w_pkt);
        }

        av_frame_free(&r_fae);
    }

    av_packet_unref(&r_pkt);
    

    for(int i = 0;i<1000;i++)
    {
        AVFrame *r_fae = NULL;
        AVPacket *w_pkt = NULL;

        if(i == 0)
        {
            /*通知解码器是结束包。*/
            int chk = abcdk_cuda_video_decode(dec_ctx, &r_fae, &r_pkt);
            if (chk < 0)
                break;
            else if(chk > 0)
            {
                chk = abcdk_cuda_video_encode(enc_ctx,&w_pkt,r_fae);
                if(chk <=0)
                    break;

                abcdk_ffmpeg_write_packet2(w,w_pkt->data,w_pkt->size,w_pkt->flags,w_stream_idx);
                av_packet_free(&w_pkt);
            }
        }
        else
        {
            int chk = abcdk_cuda_video_encode(enc_ctx,&w_pkt,NULL);
            if(chk <=0)
                break;

            abcdk_ffmpeg_write_packet2(w,w_pkt->data,w_pkt->size,w_pkt->flags,w_stream_idx);
            av_packet_free(&w_pkt);
        }
    }

    abcdk_ffmpeg_write_trailer(w);

    abcdk_cuda_video_destroy(&dec_ctx);
    abcdk_cuda_video_destroy(&enc_ctx);
    abcdk_ffmpeg_destroy(&r);
    abcdk_ffmpeg_destroy(&w);

    return 0;
}

int abcdk_test_cuda(abcdk_option_t *args)
{
    int cmd = abcdk_option_get_int(args, "--cmd", 0, 1);

    cuInit(0);

    if (cmd == 1)
        return abcdk_test_cuda_1(args);
    else if (cmd == 2)
        return abcdk_test_cuda_2(args);
    else if (cmd == 3)
        return abcdk_test_cuda_3(args);

    return 0;
}

#endif // HAVE_FFMPEG
#else  // HAVE_CUDA

int abcdk_test_cuda(abcdk_option_t *args)
{
    return 0;
}

#endif // HAVE_CUDA
