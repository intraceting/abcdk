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


int abcdk_test_torch_1(abcdk_option_t *args)
{
    int chk;

    int w = 1920 ,h = 1080;
    //int w = 1920-1 ,h = 1080-1;

    abcdk_torch_image_t *a = abcdk_torch_image_create(w, h, ABCDK_TORCH_PIXFMT_BGR24, 4);
    abcdk_torch_image_t *b = abcdk_torch_image_create(w, h, ABCDK_TORCH_PIXFMT_BGR24, 8);

    uint32_t scalar[3] = {0, 0, 255};
    abcdk_torch_imgproc_stuff(a, scalar,NULL);

    //abcdk_torch_imgcode_save("/tmp/test.cuda.a1.bmp", a);

    abcdk_torch_image_t *aa = abcdk_torch_image_clone(1,a);
    abcdk_bmp_save_file("/tmp/test.cuda.a1.bmp",aa->data[0],aa->stride[0],aa->width,-aa->height,24);
    abcdk_torch_image_free_host(&aa);

    uint32_t color[4] = {255, 0,0, 0};
    int corner[4] = {10, 10, 100, 100};

    abcdk_torch_imgproc_drawrect(a, color, 3, corner);

    // abcdk_torch_imgcode_save("/tmp/test.cuda.a2.bmp", a);
    aa = abcdk_torch_image_clone(1, a);
    abcdk_bmp_save_file("/tmp/test.cuda.a2.bmp", aa->data[0], aa->stride[0], aa->width, -aa->height, 24);
    abcdk_torch_image_free_host(&aa);

    chk = abcdk_torch_image_copy(b, a);
    assert(chk == 0);

    // abcdk_bmp_save_file("/tmp/test.cuda.a.bmp",a->data[0],a->stride[0],a->width,a->height,24);
    abcdk_torch_imgcode_save("/tmp/test.cuda.a3.jmp", a);
    abcdk_torch_imgcode_save("/tmp/test.cuda.b.bmp", b);

   // abcdk_torch_image_t *c = abcdk_torch_image_create(w, h, ABCDK_TORCH_PIXFMT_YUV420P, 567);
   // abcdk_torch_image_t *c = abcdk_torch_image_create(w, h, ABCDK_TORCH_PIXFMT_YUV422P, 567);
   // abcdk_torch_image_t *c = abcdk_torch_image_create(w, h, ABCDK_TORCH_PIXFMT_YUV444P, 567);
    //abcdk_torch_image_t *c = abcdk_torch_image_create(w, h, ABCDK_TORCH_PIXFMT_NV12, 1);
    //abcdk_torch_image_t *c = abcdk_torch_image_create(w, h, ABCDK_TORCH_PIXFMT_ARGB, 1);
    //abcdk_torch_image_t *c = abcdk_torch_image_create(w, h, ABCDK_TORCH_PIXFMT_BGR32, 1);
    abcdk_torch_image_t *c = abcdk_torch_image_create(w, h, ABCDK_TORCH_PIXFMT_RGB32, 1);
   // abcdk_torch_image_t *c = abcdk_torch_image_create(w, h, ABCDK_TORCH_PIXFMT_NV21, 2);
    //abcdk_torch_image_t *c = abcdk_torch_image_create(w, h, ABCDK_TORCH_PIXFMT_NV21, 2);
    //abcdk_torch_image_t *c = abcdk_torch_image_create(w, h, ABCDK_TORCH_PIXFMT_YUV444P10, 1);

    abcdk_torch_image_convert(c, a);

    abcdk_torch_image_t *d = abcdk_torch_image_create(w, h, ABCDK_TORCH_PIXFMT_RGB24, 1);

    abcdk_torch_image_convert(d, c);

    abcdk_torch_imgcode_save("/tmp/test.cuda.d.jmp", d);

    abcdk_torch_image_t *e = abcdk_torch_image_create(800, 600, ABCDK_TORCH_PIXFMT_RGB24, 678);

    abcdk_torch_imgproc_resize(e, NULL, d, NULL, 1, NPPI_INTER_CUBIC);

    abcdk_torch_imgcode_save("/tmp/test.cuda.e.jmp", e);

    abcdk_torch_image_t *f = abcdk_torch_image_create(800, 600, ABCDK_TORCH_PIXFMT_RGB24, 678);

    abcdk_torch_point_t dst_quad[4] = {
        {30, 30},   // 变换后的左上角
        {220, 50},  // 变换后的右上角
        {210, 220}, // 变换后的右下角
        {50, 230},  // 变换后的左下角
    };

    abcdk_torch_rect_t src_roi = {100, 100, 200, 200};

    abcdk_torch_imgproc_warp(f, NULL, dst_quad, e, NULL, NULL, 1, NPPI_INTER_CUBIC);

    abcdk_torch_imgcode_save("/tmp/test.cuda.f.jpg", f);
    // abcdk_torch_imgcode_save("/tmp/test.cuda.f2.jpg", f);

    // abcdk_torch_imgcode_save("/tmp/test.cuda.f2.bmp", f);

    abcdk_torch_image_free(&a);
    abcdk_torch_image_free(&b);
    abcdk_torch_image_free(&c);
    abcdk_torch_image_free(&d);
    abcdk_torch_image_free(&e);
    abcdk_torch_image_free(&f);

    for (int i = 0; i < 10; i++)
    {
        abcdk_torch_image_t *g = abcdk_torch_imgcode_load("/tmp/test.cuda.f.jpg");

        abcdk_torch_imgproc_drawrect(g, color, 3, corner);

        abcdk_torch_imgcode_save("/tmp/test.cuda.g2.jpg", g);

        abcdk_torch_image_free(&g);
    }

    return 0;
}

#ifdef  HAVE_FFMPEG

int abcdk_test_torch_2(abcdk_option_t *args)
{
    abcdk_ffeditor_config_t ff_r_cfg = {0};

    ff_r_cfg.file_name = abcdk_option_get(args, "--src", 0, "");
    ff_r_cfg.read_flush = abcdk_option_get_double(args, "--src-flush", 0, 0);
    ff_r_cfg.read_speed = abcdk_option_get_double(args, "--src-xpeed", 0, 1);
    ff_r_cfg.read_delay_max = abcdk_option_get_double(args, "--src-delay-max", 0, 10);
    ff_r_cfg.bit_stream_filter = 1;

    abcdk_ffeditor_t *r = abcdk_ffeditor_open(&ff_r_cfg);

    AVStream *r_video_steam = abcdk_ffeditor_find_stream(r, AVMEDIA_TYPE_VIDEO);

    abcdk_torch_vcodec_t *dec_ctx = abcdk_torch_vcodec_alloc(0);


    abcdk_torch_vcodec_param_t dec_param = {0};

    dec_param.format = abcdk_torch_vcodec_convert_from_ffmpeg(r_video_steam->codecpar->codec_id);
    dec_param.ext_data = r_video_steam->codecpar->extradata;
    dec_param.ext_size = r_video_steam->codecpar->extradata_size;

    abcdk_torch_vcodec_start(dec_ctx,&dec_param);

    AVPacket r_pkt;
    av_init_packet(&r_pkt);

    abcdk_torch_jcodec_t *jpeg_w = abcdk_torch_jcodec_alloc(1);

    abcdk_torch_jcodec_param_t jpeg_param = {0};
    jpeg_param.quality = 99;

    abcdk_torch_jcodec_start(jpeg_w,&jpeg_param);

    for (int i = 0; i < 100; i++)
    {
        int chk = abcdk_ffeditor_read_packet(r, &r_pkt, r_video_steam->index);
        if (chk < 0)
            break;

        abcdk_torch_frame_t *r_fae = NULL;
        chk = abcdk_torch_vcodec_decode_from_ffmpeg(dec_ctx, &r_fae, &r_pkt);
        if (chk < 0)
        {
            break;
        }
        else if (chk > 0)
        {
            char filename[PATH_MAX] = {0};
            sprintf(filename, "/tmp/ddd/%06d.jpg", r_fae->pts);

            abcdk_mkdir(filename, 0755);

            abcdk_torch_jcodec_encode_to_file(jpeg_w, filename, r_fae->img);
        }

        abcdk_torch_frame_free(&r_fae);
    }

    abcdk_torch_jcodec_free(&jpeg_w);

    av_packet_unref(&r_pkt);

    abcdk_torch_vcodec_free(&dec_ctx);
    abcdk_ffeditor_destroy(&r);

    return 0;
}

int abcdk_test_torch_3(abcdk_option_t *args)
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

    abcdk_torch_vcodec_t *dec_ctx = abcdk_torch_vcodec_alloc(0);
    abcdk_torch_vcodec_t *enc_ctx = abcdk_torch_vcodec_alloc(1);


    abcdk_torch_vcodec_param_t dec_param = {0},enc_param = {0};

    dec_param.format = abcdk_torch_vcodec_convert_from_ffmpeg(r_video_steam->codecpar->codec_id);
    dec_param.ext_data = r_video_steam->codecpar->extradata;
    dec_param.ext_size = r_video_steam->codecpar->extradata_size;

    abcdk_torch_vcodec_start(dec_ctx,&dec_param);
  
    enc_param.format = abcdk_torch_vcodec_convert_from_ffmpeg(AV_CODEC_ID_H264);
    enc_param.fps_d = 1;
    enc_param.fps_n = 25;
    enc_param.width = r_video_steam->codecpar->width;
    enc_param.height = r_video_steam->codecpar->height;
    enc_param.bitrate = 15000 * 1000;
    enc_param.peak_bitrate = 15000 * 1000;

    abcdk_torch_vcodec_start(enc_ctx,&enc_param);

    AVCodecContext *enc_opt = abcdk_avcodec_alloc3(AV_CODEC_ID_H264, 1);

    abcdk_avcodec_encode_video_fill_time_base(enc_opt, 25);

    enc_opt->width = r_video_steam->codecpar->width;
    enc_opt->height = r_video_steam->codecpar->height;
    enc_opt->extradata = (uint8_t*)av_memdup(enc_param.ext_data, enc_param.ext_size);
    enc_opt->extradata_size = enc_param.ext_size;
    enc_opt->max_b_frames = enc_param.max_b_frames;
    enc_opt->bit_rate_tolerance = enc_param.peak_bitrate;
    enc_opt->bit_rate = enc_param.bitrate ;

    int w_stream_idx = abcdk_ffeditor_add_stream(w, enc_opt, 1);

    abcdk_avcodec_free(&enc_opt);

    abcdk_ffeditor_write_header(w, 1);

    AVPacket r_pkt, *w_pkt = NULL;
    av_init_packet(&r_pkt);

    for (int i = 0; i < 10000; i++)
    {
        int chk = abcdk_ffeditor_read_packet(r, &r_pkt, r_video_steam->index);
        if (chk < 0)
            break;

        abcdk_torch_frame_t *r_fae = NULL;
        chk = abcdk_torch_vcodec_decode_from_ffmpeg(dec_ctx, &r_fae, &r_pkt);
        if (chk < 0)
        {
            break;
        }
        else if (chk > 0)
        {
            AVPacket *w_pkt = NULL;
            int chk = abcdk_torch_vcodec_encode_to_ffmpeg(enc_ctx, &w_pkt, r_fae);
            if (chk < 0)
                break;
            else if( chk > 0)
            {
                abcdk_ffeditor_write_packet2(w, w_pkt->data, w_pkt->size, w_pkt->flags, w_stream_idx);
                av_packet_free(&w_pkt);
            }
        }

        abcdk_torch_frame_free(&r_fae);
    }

    av_packet_unref(&r_pkt);

    for (int i = 0; i < 100; i++)
    {
        abcdk_torch_frame_t *r_fae = NULL;
        AVPacket *w_pkt = NULL;

        /*通知解码器是结束包。*/
        int chk = abcdk_torch_vcodec_decode_from_ffmpeg(dec_ctx, &r_fae, (i == 0 ? &r_pkt : NULL));
        if (chk < 0)
            break;
        else //if (chk > 0)
        {
            chk = abcdk_torch_vcodec_encode_to_ffmpeg(enc_ctx, &w_pkt, r_fae);
            if (chk <= 0)
                break;

            abcdk_ffeditor_write_packet2(w, w_pkt->data, w_pkt->size, w_pkt->flags, w_stream_idx);
            av_packet_free(&w_pkt);
        }

        abcdk_torch_frame_free(&r_fae);
    }

    abcdk_ffeditor_write_trailer(w);

    
    abcdk_torch_vcodec_free(&dec_ctx);
    abcdk_torch_vcodec_free(&enc_ctx);
    
    abcdk_ffeditor_destroy(&r);
    abcdk_ffeditor_destroy(&w);

    return 0;
}

#endif //HAVE_FFMPEG

int abcdk_test_torch_4(abcdk_option_t *args)
{
    int n = 1, w = 300, h = 300 , depth =3;

   // abcdk_trace_printf(LOG_WARNING,_("哈哈哈"));

}

int abcdk_test_torch(abcdk_option_t *args)
{
    int cmd = abcdk_option_get_int(args, "--cmd", 0, 1);

    int gpu = abcdk_option_get_int(args, "--gpu", 0, 0);

    char name[256] = {0};
    int chk = abcdk_torch_get_device_name(name, gpu);
    assert(chk == 0);

    fprintf(stderr, "%s\n", name);

    abcdk_torch_context_t *torch_ctx = abcdk_torch_context_create(gpu, 0);

    abcdk_torch_context_current_set(torch_ctx);


    if (cmd == 1)
        return abcdk_test_torch_1(args);
#ifdef HAVE_FFMPEG
    else if (cmd == 2)
        return abcdk_test_torch_2(args);
    else if (cmd == 3)
        return abcdk_test_torch_3(args);
#endif //HAVE_FFMPEG
    else if (cmd == 4)
        return abcdk_test_torch_4(args);


    abcdk_torch_context_current_set(NULL);
    abcdk_torch_context_destroy(&torch_ctx);

    return 0;
}

