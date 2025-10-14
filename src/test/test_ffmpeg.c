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

    rd_param.url = abcdk_option_get(args,"--src-fmt",0,"");
    rd_param.url = abcdk_option_get(args,"--src",0,"");
    rd_param.read_speed_scale = abcdk_option_get_int(args,"--src-xpeed",0,1000);
    rd_param.read_mp4toannexb = abcdk_option_get_int(args,"--src-mp4toannexb ",0,1);
    rd_param.read_ignore_video = abcdk_option_get_int(args,"--src-ignore-video",0,0);
    rd_param.read_ignore_audio = abcdk_option_get_int(args,"--src-ignore-audio",0,0);
    rd_param.read_ignore_subtitle = abcdk_option_get_int(args,"--src-ignore-subtitle ",0,0);
    wr_param.fmt = abcdk_option_get(args,"--dst-fmt",0,"");
    wr_param.url = abcdk_option_get(args,"--dst",0,"");

    
    abcdk_ffmpeg_editor_t *rd_ctx = abcdk_ffmpeg_editor_alloc(0);
    abcdk_ffmpeg_editor_t *wr_ctx = abcdk_ffmpeg_editor_alloc(1);

    chk = abcdk_ffmpeg_editor_open(rd_ctx,&rd_param);
    assert(chk == 0);
    chk = abcdk_ffmpeg_editor_open(wr_ctx,&wr_param);
    assert(chk == 0);

    AVPacket *rd_pkt = av_packet_alloc();

    for(int i = 0;i<1000;i++)
    {
        av_packet_unref(rd_pkt);
        chk = abcdk_ffmpeg_editor_read_packet(rd_ctx,rd_pkt);
        if(chk != 0)
            break;

        double dts_sec = abcdk_ffmpeg_editor_ts2sec(rd_ctx,rd_pkt->stream_index,rd_pkt->dts);
        double pts_sec = abcdk_ffmpeg_editor_ts2sec(rd_ctx,rd_pkt->stream_index,rd_pkt->pts);

        abcdk_trace_printf(LOG_DEBUG,"stream(%d),dts(%0.3f),pts(%0.3f) ",rd_pkt->stream_index,dts_sec,pts_sec);
    }

    av_packet_free(&rd_pkt);

    abcdk_ffmpeg_editor_free(&rd_ctx);
    abcdk_ffmpeg_editor_free(&wr_ctx);

    return 0;
}

int abcdk_test_codec(abcdk_option_t *args)
{
    int chk;
    abcdk_ffmpeg_editor_param_t rd_param = {0};
    abcdk_ffmpeg_editor_param_t wr_param = {0};

    rd_param.url = abcdk_option_get(args,"--src-fmt",0,"");
    rd_param.url = abcdk_option_get(args,"--src",0,"");
    rd_param.read_speed_scale = abcdk_option_get_int(args,"--src-xpeed",0,1000);
    rd_param.read_mp4toannexb = abcdk_option_get_int(args,"--src-mp4toannexb ",0,1);
    wr_param.fmt = abcdk_option_get(args,"--dst-fmt",0,"");
    wr_param.url = abcdk_option_get(args,"--dst",0,"");

    
    abcdk_ffmpeg_editor_t *rd_ctx = abcdk_ffmpeg_editor_alloc(0);
    abcdk_ffmpeg_editor_t *wr_ctx = abcdk_ffmpeg_editor_alloc(1);

    chk = abcdk_ffmpeg_editor_open(rd_ctx,&rd_param);
    assert(chk == 0);
    chk = abcdk_ffmpeg_editor_open(wr_ctx,&wr_param);
    assert(chk == 0);

    AVFrame *rd_fae = av_frame_alloc();

    for (int i = 0; i < 1000; i++)
    {
        av_frame_unref(rd_fae);
   //     chk = abcdk_ffmpeg_editor_read_frame(rd_ctx, rd_fae);
        if (chk != 0)
            break;

    //    double pts_sec = abcdk_ffmpeg_editor_ts2sec(rd_ctx, rd_fae->stream_index, rd_fae->pts);

   //     abcdk_trace_printf(LOG_DEBUG, "WH(%dx%d),format(%d),pts(%0.3f) ", rd_fae->width, rd_fae->height, rd_fae->format, pts_sec);
    }

    av_frame_free(&rd_fae);

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

#endif //HAVE_FFMPEG

int abcdk_test_ffmpeg(abcdk_option_t *args)
{
#ifdef HAVE_FFMPEG

    int cmd = abcdk_option_get_int(args,"--cmd",0,1);

    if(cmd == 1)
        abcdk_test_record(args);
    else if(cmd == 2)
        abcdk_test_codec(args);
    else if(cmd == 3)
        abcdk_test_extradata(args);
    else if(cmd == 4)
        abcdk_test_audio(args);
    else if(cmd == 5)
        abcdk_test_record2(args);

#endif //HAVE_FFMPEG

    return 0;
}
