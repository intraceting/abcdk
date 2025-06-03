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

#ifdef HAVE_LIVE555 

int abcdk_test_server(abcdk_option_t *args)
{

#ifdef HAVE_FFMPEG

    abcdk_rtsp_server_t *ctx = abcdk_rtsp_server_create(12345, 0x01 | 0x02);

    int chk = abcdk_rtsp_server_set_auth(ctx, "haha");
    chk = abcdk_rtsp_server_set_auth(ctx, "hehe");
    assert(chk == 0);

    const char *cert_p = abcdk_option_get(args, "--cert", 0, NULL);
    const char *key_p = abcdk_option_get(args, "--key", 0, NULL);

    if (cert_p && key_p)
        abcdk_rtsp_server_set_tls(ctx, cert_p, key_p, 0, 0);

    chk = abcdk_rtsp_server_start(ctx);
    assert(chk == 0);

    chk = abcdk_rtsp_server_add_user(ctx, "cccc", "aaaa",ABCDK_RTSP_AUTH_NORMAL,0,0);
    assert(chk == 0);
    chk = abcdk_rtsp_server_add_user(ctx, "dddd", "bbbb",ABCDK_RTSP_AUTH_NORMAL,0,0);
    assert(chk == 0);

    abcdk_rtsp_server_remove_user(ctx, "cccc");
    abcdk_rtsp_server_remove_user(ctx, "dddd");

    chk = abcdk_rtsp_server_add_user(ctx, "aaaa", "aaaa",ABCDK_RTSP_AUTH_NORMAL,0,0);
    assert(chk == 0);
    chk = abcdk_rtsp_server_add_user(ctx, "aaaa", "bbbb",ABCDK_RTSP_AUTH_NORMAL,0,0);
    assert(chk == 0);

    const char *name = "aaa";

    chk = abcdk_rtsp_server_create_media(ctx, name, NULL, NULL);
    assert(chk == 0);

    abcdk_ffeditor_config_t rcfg = {0};

    rcfg.url = abcdk_option_get(args, "--src", 0, "");
    rcfg.read_speed = abcdk_option_get_double(args, "--src-xpeed", 0, 1);
    rcfg.read_delay_max = abcdk_option_get_double(args, "--src-delay-max", 0, 1);
    rcfg.bit_stream_filter = 1;

    abcdk_ffeditor_t *r = abcdk_ffeditor_open(&rcfg);

    AVFormatContext *rf = abcdk_ffeditor_ctxptr(r);

    // abcdk_avformat_dump(rf,0);

    int stream[16] = {-1, -1, -1, -1};

    for (int i = 0; i < abcdk_ffeditor_streams(r); i++)
    {
        AVStream *p = abcdk_ffeditor_streamptr(r, i);

        if (p->codecpar->codec_id == AV_CODEC_ID_HEVC)
        {
            abcdk_object_t *extdata = abcdk_object_copyfrom(p->codecpar->extradata, p->codecpar->extradata_size);
            stream[i] = abcdk_rtsp_server_add_stream(ctx, name, ABCDK_RTSP_CODEC_H265, extdata, (p->codecpar->bit_rate > 0 ? p->codecpar->bit_rate / 1000 : 3000), 10);
            abcdk_object_unref(&extdata);
        }
        else if (p->codecpar->codec_id == AV_CODEC_ID_H264)
        {
            abcdk_object_t *extdata = abcdk_object_copyfrom(p->codecpar->extradata, p->codecpar->extradata_size);
            stream[i] = abcdk_rtsp_server_add_stream(ctx, name, ABCDK_RTSP_CODEC_H264, extdata, (p->codecpar->bit_rate > 0 ? p->codecpar->bit_rate / 1000 : 3000), 10);
            abcdk_object_unref(&extdata);
        }
        else if (p->codecpar->codec_id == AV_CODEC_ID_AAC)
        {
            abcdk_object_t *extdata = abcdk_object_copyfrom(p->codecpar->extradata, p->codecpar->extradata_size);
            stream[i] = abcdk_rtsp_server_add_stream(ctx, name, ABCDK_RTSP_CODEC_AAC, extdata, (p->codecpar->bit_rate > 0 ? p->codecpar->bit_rate / 1000 : 96), 10);
            abcdk_object_unref(&extdata);
        }
        else if (p->codecpar->codec_id == AV_CODEC_ID_PCM_MULAW)
        {
            abcdk_object_t *extdata = abcdk_object_alloc3(sizeof(int), 2); //[0] = channels,[1]=sample_rate
            ABCDK_PTR2I32(extdata->pptrs[0], 0) = p->codecpar->channels;
            ABCDK_PTR2I32(extdata->pptrs[1], 0) = p->codecpar->sample_rate;
            stream[i] = abcdk_rtsp_server_add_stream(ctx, name, ABCDK_RTSP_CODEC_G711U, extdata, (p->codecpar->bit_rate > 0 ? p->codecpar->bit_rate / 1000 : 96), 10);
            abcdk_object_unref(&extdata);
        }
        else if (p->codecpar->codec_id == AV_CODEC_ID_PCM_ALAW)
        {
            abcdk_object_t *extdata = abcdk_object_alloc3(sizeof(int), 2); //[0] = channels,[1]=sample_rate
            ABCDK_PTR2I32(extdata->pptrs[0], 0) = p->codecpar->channels;
            ABCDK_PTR2I32(extdata->pptrs[1], 0) = p->codecpar->sample_rate;
            stream[i] = abcdk_rtsp_server_add_stream(ctx, name, ABCDK_RTSP_CODEC_G711A, extdata, (p->codecpar->bit_rate > 0 ? p->codecpar->bit_rate / 1000 : 96), 10);
            abcdk_object_unref(&extdata);
        }
        else if (p->codecpar->codec_id == AV_CODEC_ID_OPUS)
        {
            abcdk_object_t *extdata = abcdk_object_alloc3(sizeof(int), 2); //[0] = channels,[1]=sample_rate
            ABCDK_PTR2I32(extdata->pptrs[0], 0) = p->codecpar->channels;
            ABCDK_PTR2I32(extdata->pptrs[1], 0) = p->codecpar->sample_rate;
            stream[i] = abcdk_rtsp_server_add_stream(ctx, name, ABCDK_RTSP_CODEC_OPUS, extdata, (p->codecpar->bit_rate > 0 ? p->codecpar->bit_rate / 1000 : 96), 10);
            abcdk_object_unref(&extdata);
        }
    }

    abcdk_rtsp_server_play_media(ctx, name);
    //  abcdk_rtsp_server_remove_media(ctx,name);

    //  goto END;

    // abcdk_rtsp_server_start(ctx);

    AVPacket pkt;

    av_init_packet(&pkt);
    for (int i = 0; i < 10000; i++)
    {
        int n = abcdk_ffeditor_read_packet(r, &pkt, -1);
        if (n < 0)
            break;

        double pts_sec = abcdk_ffeditor_ts2sec(r, pkt.stream_index, pkt.pts);                     //// 秒。
        double dur_sec = (double)pkt.duration * abcdk_ffeditor_timebase_q2d(r, pkt.stream_index); // 秒。

        //  abcdk_trace_printf(LOG_DEBUG, "IDX(%d), DTS(%lld),PTS(%lld,%.6f),DUR(%.6f),", pkt.stream_index,pkt.dts, pkt.pts,pts_sec, dur_sec);

        abcdk_rtsp_server_play_stream(ctx, name, stream[pkt.stream_index], pkt.data, pkt.size, pts_sec * 1000000, dur_sec * 1000000); // 转微秒。
    }

    av_packet_unref(&pkt);

    abcdk_rtsp_server_remove_media(ctx, name);

END:

    abcdk_ffeditor_destroy(&r);

    abcdk_rtsp_server_stop(ctx);

    abcdk_rtsp_server_destroy(&ctx);

#endif //HAVE_FFMPEG

    return 0;
}

int abcdk_test_relay(abcdk_option_t *args)
{
    int chk;

    int port = abcdk_option_get_int(args, "--port", 0, 12345);

    abcdk_rtsp_server_t *server_ctx = abcdk_rtsp_server_create(port, 0x01 | 0x02| 0x10);

    chk = abcdk_rtsp_server_set_auth(server_ctx, "haha");
    assert(chk == 0);

    chk = abcdk_rtsp_server_start(server_ctx);
    assert(chk == 0);

    int totp_scheme = abcdk_option_get_int(args, "--totp-scheme", 0, ABCDK_RTSP_AUTH_TOTP_SHA1);
    int totp_time_step = abcdk_option_get_int(args, "--totp-time-step", 0, 30);
    int totp_digit_size = abcdk_option_get_int(args, "--totp-digit-size", 0, 6);

    chk = abcdk_rtsp_server_add_user(server_ctx, "aaaa", "bbbb",ABCDK_RTSP_AUTH_NORMAL,0,0);
    assert(chk == 0);

    chk = abcdk_rtsp_server_add_user(server_ctx, "bbbb", "12345678901234567890",totp_scheme,totp_time_step,totp_digit_size);
    assert(chk == 0);

    abcdk_rtsp_relay_t *relay_ctx[100] = {0};

    for (int i = 0; i < 100; i++)
    {
        const char *src_p = abcdk_option_get(args, "--src", i, NULL);
        if (!src_p)
            break;

        char name[NAME_MAX] = {0};

        sprintf(name, "relay%d", i + 1);

        relay_ctx[i] = abcdk_rtsp_relay_create(server_ctx, name, src_p, NULL, 1, 5, 5);
    }

    fprintf(stderr, "\npress q key to exit.\n");
    while (getchar() != 'q')
        ;

    for (int i = 0; i < 100; i++)
    {
        abcdk_rtsp_relay_destroy(&relay_ctx[i]);
    }

    abcdk_rtsp_server_stop(server_ctx);

    abcdk_rtsp_server_destroy(&server_ctx);

    return 0;
}

int abcdk_test_rtspserver(abcdk_option_t *args)
{
    int cmd = abcdk_option_get_int(args, "--cmd", 0, 1);

    if (cmd == 1)
        abcdk_test_server(args);
    if (cmd == 2)
        abcdk_test_relay(args);

    return 0;
}

#else // HAVE_LIVE555

int abcdk_test_rtspserver(abcdk_option_t *args)
{
    return 0;
}

#endif // HAVE_LIVE555