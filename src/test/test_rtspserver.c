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

int abcdk_test_rtspserver(abcdk_option_t *args)
{

    abcdk_rtsp_server_t *ctx = abcdk_rtsp_server_create(12345,"ABCDK");

    abcdk_rtsp_server_start(ctx);


    int media = abcdk_rtsp_server_create_media(ctx,"aaa.mp4","haha","haha test");
    assert(media > 0);

    abcdk_ffeditor_config_t rcfg = {0};

    rcfg.file_name = abcdk_option_get(args,"--src",0,"");
    rcfg.read_speed = abcdk_option_get_double(args,"--src-xpeed",0,1);
    rcfg.read_delay_max = abcdk_option_get_double(args,"--src-delay-max",0,1);
    rcfg.bit_stream_filter = 1;

    abcdk_ffeditor_t *r = abcdk_ffeditor_open(&rcfg);

    AVFormatContext *rf = abcdk_ffeditor_ctxptr(r);

    abcdk_avformat_dump(rf,0);

    int stream = -1;
    
    for(int i = 0;i<abcdk_ffeditor_streams(r);i++)
    {
        AVStream * p = abcdk_ffeditor_streamptr(r,i);

        if(p->codecpar->codec_id == AV_CODEC_ID_HEVC)
        {   
            abcdk_hevc_extradata_t extradata = {0};
            abcdk_hevc_extradata_deserialize(p->codecpar->extradata, p->codecpar->extradata_size, &extradata);

            abcdk_object_t *extdata = abcdk_object_copyfrom(p->codecpar->extradata,p->codecpar->extradata_size);
            stream = abcdk_rtsp_server_media_add_stream(ctx,media,ABCDK_RTSP_CODEC_H265,extdata,100);
            abcdk_object_unref(&extdata);


        }
        else if(p->codecpar->codec_id == AV_CODEC_ID_H264)
        {
            abcdk_object_t *extdata = abcdk_object_copyfrom(p->codecpar->extradata,p->codecpar->extradata_size);
            stream = abcdk_rtsp_server_media_add_stream(ctx,media,ABCDK_RTSP_CODEC_H264,extdata,100);
            abcdk_object_unref(&extdata);
        }
    }

    abcdk_rtsp_server_media_play(ctx,media);

    AVPacket pkt;

    av_init_packet(&pkt);
    for (int i = 0; i < 1000000; i++)
    {
        int n = abcdk_ffeditor_read_packet(r, &pkt, -1);
        if (n < 0)
            break;

        abcdk_rtsp_server_media_append_stream(ctx, media, stream, pkt.data + 4, pkt.size - 4, pkt.dts, pkt.pts, pkt.duration * 1000);
    }

    abcdk_rtsp_server_stop(ctx);

    abcdk_rtsp_server_destroy(&ctx);

    return 0;
}

#else // HAVE_LIVE555

int abcdk_test_rtspserver(abcdk_option_t *args)
{
    return 0;
}

#endif // HAVE_LIVE555