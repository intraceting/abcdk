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

#ifdef HAVE_FFMPEG
int abcdk_test_record(abcdk_option_t *args)
{
    int chk;
    abcdk_ffmpeg_editor_param_t rd_param = {0};
    abcdk_ffmpeg_editor_param_t wr_param = {0};

    rd_param.url = abcdk_option_get(args, "--src-fmt", 0, "");
    rd_param.url = abcdk_option_get(args, "--src", 0, "");
    rd_param.read_nodelay = abcdk_option_get_int(args, "--src-nodelay", 0, 0);
    rd_param.read_rate_scale = abcdk_option_get_int(args, "--src-rate-scale", 0, 1000);
    rd_param.read_mp4toannexb = abcdk_option_get_int(args, "--src-mp4toannexb", 0, 1);
    rd_param.read_ignore_video = abcdk_option_get_int(args, "--src-ignore-video", 0, 0);
    rd_param.read_ignore_audio = abcdk_option_get_int(args, "--src-ignore-audio", 0, 0);
    rd_param.read_ignore_subtitle = abcdk_option_get_int(args, "--src-ignore-subtitle", 0, 0);
    wr_param.fmt = abcdk_option_get(args, "--dst-fmt", 0, "");
    wr_param.url = abcdk_option_get(args, "--dst", 0, "");

    abcdk_ffmpeg_editor_t *rd_ctx = abcdk_ffmpeg_editor_alloc(0);
    abcdk_ffmpeg_editor_t *wr_ctx = abcdk_ffmpeg_editor_alloc(1);

    chk = abcdk_ffmpeg_editor_open(rd_ctx, &rd_param);
    assert(chk == 0);

    chk = abcdk_ffmpeg_editor_open(wr_ctx, &wr_param);
    assert(chk == 0);

    for (int i = 0; i < abcdk_ffmpeg_editor_stream_nb(rd_ctx); i++)
    {
        AVStream *p = abcdk_ffmpeg_editor_stream_ctx(rd_ctx, i);

#if 0
#ifdef FF_API_LAVF_AVCTX
        chk = abcdk_ffmpeg_editor_add_stream(wr_ctx,p->codec);
        assert(chk == i);
#endif // #ifdef FF_API_LAVF_AVCTX
#else
        chk = abcdk_ffmpeg_editor_add_stream2(wr_ctx, p->codecpar, &p->time_base, &p->avg_frame_rate, &p->r_frame_rate);
        assert(chk == i);
#endif
    }

    // chk = abcdk_ffmpeg_editor_write_header(wr_ctx);
    // assert(chk == 0);

    AVPacket *rd_pkt = av_packet_alloc();

    for (int i = 0; i < 10000000; i++)
    {
        chk = abcdk_ffmpeg_editor_read_packet(rd_ctx, rd_pkt);
        if (chk != 0)
            break;

        double dts_sec = abcdk_ffmpeg_editor_stream_ts2sec(rd_ctx, rd_pkt->stream_index, rd_pkt->dts);
        double pts_sec = abcdk_ffmpeg_editor_stream_ts2sec(rd_ctx, rd_pkt->stream_index, rd_pkt->pts);

        abcdk_trace_printf(LOG_DEBUG, "stream(%d),dts(%0.3f),pts(%0.3f) ", rd_pkt->stream_index, dts_sec, pts_sec);

        AVStream *p = abcdk_ffmpeg_editor_stream_ctx(wr_ctx, rd_pkt->stream_index);

        //        AVRational time_base = p->time_base;

        //   chk = abcdk_ffmpeg_editor_write_packet(wr_ctx,rd_pkt,&p->time_base);
        chk = abcdk_ffmpeg_editor_write_packet(wr_ctx, rd_pkt);
        if (chk != 0)
            break;
    }

    av_packet_free(&rd_pkt);

    // chk = abcdk_ffmpeg_editor_write_trailer(wr_ctx);
    // assert(chk == 0);

    abcdk_ffmpeg_editor_free(&rd_ctx);
    abcdk_ffmpeg_editor_free(&wr_ctx);

    return 0;
}

static void _abcdk_test_codec_fill_enc_params(AVCodecParameters *enc_param, int width, int height, int bitrate)
{
    // 设置编码类型: 视频流
    enc_param->codec_type = AVMEDIA_TYPE_VIDEO;

    // 设置编码ID为 H.265 (HEVC)
    enc_param->codec_id = AV_CODEC_ID_H264;

    // 设置分辨率
    enc_param->width = width;
    enc_param->height = height;

    // 设置帧率(通常与 time_base 成反比)
    enc_param->sample_aspect_ratio = (AVRational){1, 1};

    // 设置像素格式 (例如 YUV420P)
    enc_param->format = AV_PIX_FMT_YUV420P;

    // 设置码率(单位: bit/s)
    enc_param->bit_rate = bitrate;

    // 可选: 时间基(有时需要在 AVStream 设置)
     enc_param->video_delay = 10;

    // 如果是 HDR 或 10-bit, 可以设置如下: 
    // enc_param->format = AV_PIX_FMT_YUV420P10LE;

    // 可选: 颜色空间信息
    // enc_param->color_primaries = AVCOL_PRI_BT709;
    // enc_param->color_trc = AVCOL_TRC_BT709;
    // enc_param->colorspace = AVCOL_SPC_BT709;
}

int abcdk_test_codec(abcdk_option_t *args)
{

    int chk;

    abcdk_ffmpeg_editor_param_t rd_param = {0};
    abcdk_ffmpeg_editor_param_t wr_param = {0};

    rd_param.url = abcdk_option_get(args, "--src-fmt", 0, "");
    rd_param.url = abcdk_option_get(args, "--src", 0, "");
    rd_param.read_nodelay = abcdk_option_get_int(args, "--src-nodelay", 0, 0);
    rd_param.read_rate_scale = abcdk_option_get_int(args, "--src-rate-scale", 0, 1000);
    rd_param.read_mp4toannexb = abcdk_option_get_int(args, "--src-mp4toannexb", 0, 1);
    rd_param.read_ignore_video = abcdk_option_get_int(args, "--src-ignore-video", 0, 0);
    rd_param.read_ignore_audio = abcdk_option_get_int(args, "--src-ignore-audio", 0, 0);
    rd_param.read_ignore_subtitle = abcdk_option_get_int(args, "--src-ignore-subtitle", 0, 0);
    wr_param.fmt = abcdk_option_get(args, "--dst-fmt", 0, "");
    wr_param.url = abcdk_option_get(args, "--dst", 0, "");

    abcdk_ffmpeg_editor_t *rd_ctx = abcdk_ffmpeg_editor_alloc(0);
    abcdk_ffmpeg_editor_t *wr_ctx = abcdk_ffmpeg_editor_alloc(1);

    chk = abcdk_ffmpeg_editor_open(rd_ctx, &rd_param);
    assert(chk == 0);

    chk = abcdk_ffmpeg_editor_open(wr_ctx, &wr_param);
    assert(chk == 0);

    abcdk_ffmpeg_decoder_t *dec_ctx;
    abcdk_ffmpeg_encoder_t *enc_ctx;

    

    for (int i = 0; i < abcdk_ffmpeg_editor_stream_nb(rd_ctx); i++)
    {
        AVStream *p = abcdk_ffmpeg_editor_stream_ctx(rd_ctx, i);

        if (p->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        {

            dec_ctx = abcdk_ffmpeg_decoder_alloc3(p->codecpar->codec_id);
            abcdk_ffmpeg_decoder_init(dec_ctx, p->codecpar);
            abcdk_ffmpeg_decoder_open(dec_ctx, NULL);

            AVCodecParameters *enc_param = avcodec_parameters_alloc();

            _abcdk_test_codec_fill_enc_params(enc_param, p->codecpar->width, p->codecpar->height, 5000 * 1000);

            AVRational time_base = av_make_q(1, 130);
            
            enc_ctx = abcdk_ffmpeg_encoder_alloc3(enc_param->codec_id);
            abcdk_ffmpeg_encoder_init(enc_ctx, enc_param, &time_base, NULL);
            abcdk_ffmpeg_encoder_open(enc_ctx, NULL);

            abcdk_ffmpeg_encoder_get_param(enc_ctx, enc_param);

            chk = abcdk_ffmpeg_editor_add_stream2(wr_ctx, enc_param, &time_base, NULL, NULL);
            assert(chk >= 0);

            avcodec_parameters_free(&enc_param);

            break;
        }
    }

    chk = abcdk_ffmpeg_editor_write_header(wr_ctx);
    assert(chk == 0);

    AVPacket *rd_pkt = av_packet_alloc();
    AVFrame *rd_fea = av_frame_alloc();

    for (int i = 0; i < 10000000; i++)
    {
        chk = abcdk_ffmpeg_editor_read_packet(rd_ctx, rd_pkt);
        if (chk != 0)
            break;

        chk = abcdk_ffmpeg_decoder_send(dec_ctx, rd_pkt);
        assert(chk == 0);

        chk = abcdk_ffmpeg_decoder_recv(dec_ctx, rd_fea);
        if (chk == 0)
            continue;
        else if (chk < 0)
            break;

        chk = abcdk_ffmpeg_encoder_send(enc_ctx, rd_fea);
        assert(chk == 0);

        chk = abcdk_ffmpeg_encoder_recv(enc_ctx, rd_pkt);
        if (chk == 0)
            continue;
        else if (chk < 0)
            break;

        chk = abcdk_ffmpeg_editor_write_packet(wr_ctx, rd_pkt);
        if (chk != 0)
            break;
    }

    chk = abcdk_ffmpeg_decoder_send(dec_ctx, NULL);
    assert(chk == 0);

    for (int i = 0; i < 10000000; i++)
    {
        chk = abcdk_ffmpeg_decoder_recv(dec_ctx, rd_fea);
        if (chk > 0)
        {
            chk = abcdk_ffmpeg_encoder_send(enc_ctx, rd_fea);
            assert(chk == 0);
        }
        else
        {
            chk = abcdk_ffmpeg_encoder_send(enc_ctx, NULL);
            assert(chk == 0);
        }

        chk = abcdk_ffmpeg_encoder_recv(enc_ctx, rd_pkt);
        if (chk == 0)
            continue;
        else if (chk < 0)
            break;

        chk = abcdk_ffmpeg_editor_write_packet(wr_ctx, rd_pkt);
        if (chk != 0)
            break;
    }

    av_frame_free(&rd_fea);
    av_packet_free(&rd_pkt);

    abcdk_ffmpeg_decoder_free(&dec_ctx);
    abcdk_ffmpeg_encoder_free(&enc_ctx);

    abcdk_ffmpeg_editor_free(&rd_ctx);
    abcdk_ffmpeg_editor_free(&wr_ctx);

    return 0;
}

int abcdk_test_extradata(abcdk_option_t *args)
{
    return 0;
}

int abcdk_test_audio(abcdk_option_t *args)
{
    return 0;
}

int abcdk_test_record2(abcdk_option_t *args)
{

    return 0;
}

#endif // HAVE_FFMPEG

int abcdk_test_ffmpeg(abcdk_option_t *args)
{
#ifdef HAVE_FFMPEG

    int cmd = abcdk_option_get_int(args, "--cmd", 0, 1);

    if (cmd == 1)
        abcdk_test_record(args);
    else if (cmd == 2)
        abcdk_test_codec(args);
    else if (cmd == 3)
        abcdk_test_extradata(args);
    else if (cmd == 4)
        abcdk_test_audio(args);
    else if (cmd == 5)
        abcdk_test_record2(args);

#endif // HAVE_FFMPEG

    return 0;
}
