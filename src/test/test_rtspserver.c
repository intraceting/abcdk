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

    abcdk_rtsp_server_t *ctx = abcdk_rtsp_server_create(12345,0x01|0x02);

    int chk = abcdk_rtsp_server_set_auth(ctx,"haha");
    chk = abcdk_rtsp_server_set_auth(ctx,"hehe");
    assert(chk == 0);

    chk = abcdk_rtsp_server_start(ctx);
    assert(chk == 0);

    chk = abcdk_rtsp_server_add_user(ctx,"cccc","aaaa");
    assert(chk == 0);
    chk = abcdk_rtsp_server_add_user(ctx,"dddd","bbbb");
    assert(chk == 0);

    abcdk_rtsp_server_remove_user(ctx,"cccc");
    abcdk_rtsp_server_remove_user(ctx,"dddd");

    chk = abcdk_rtsp_server_add_user(ctx,"aaaa","aaaa");
    assert(chk == 0);
    chk = abcdk_rtsp_server_add_user(ctx,"aaaa","bbbb");
    assert(chk == 0);


    int media = abcdk_rtsp_server_create_media(ctx,"aaa",NULL,NULL);
    assert(media > 0);

    abcdk_ffeditor_config_t rcfg = {0};

    rcfg.file_name = abcdk_option_get(args,"--src",0,"");
    rcfg.read_speed = abcdk_option_get_double(args,"--src-xpeed",0,1);
    rcfg.read_delay_max = abcdk_option_get_double(args,"--src-delay-max",0,1);
    rcfg.bit_stream_filter = 1;

    abcdk_ffeditor_t *r = abcdk_ffeditor_open(&rcfg);

    AVFormatContext *rf = abcdk_ffeditor_ctxptr(r);

    //abcdk_avformat_dump(rf,0);

    int stream[16] = {-1,-1,-1,-1};
    
    for(int i = 0;i<abcdk_ffeditor_streams(r);i++)
    {
        AVStream * p = abcdk_ffeditor_streamptr(r,i);

        if(p->codecpar->codec_id == AV_CODEC_ID_HEVC)
        {   
            abcdk_object_t *extdata = abcdk_object_copyfrom(p->codecpar->extradata,p->codecpar->extradata_size);
            stream[i] = abcdk_rtsp_server_add_stream(ctx,media,ABCDK_RTSP_CODEC_H265,extdata,10);
            abcdk_object_unref(&extdata);


        }
        else if(p->codecpar->codec_id == AV_CODEC_ID_H264)
        {
            abcdk_object_t *extdata = abcdk_object_copyfrom(p->codecpar->extradata,p->codecpar->extradata_size);
            stream[i] = abcdk_rtsp_server_add_stream(ctx,media,ABCDK_RTSP_CODEC_H264,extdata,10);
            abcdk_object_unref(&extdata);
        }
        else if(p->codecpar->codec_id == AV_CODEC_ID_AAC)
        {
            abcdk_object_t *extdata = abcdk_object_copyfrom(p->codecpar->extradata,p->codecpar->extradata_size);
            stream[i] = abcdk_rtsp_server_add_stream(ctx,media,ABCDK_RTSP_CODEC_AAC,extdata,10);
            abcdk_object_unref(&extdata);
        }
    }

   abcdk_rtsp_server_play_media(ctx,media);
 //  abcdk_rtsp_server_remove_media(ctx,media);

 //  goto END;

   //abcdk_rtsp_server_start(ctx);

    AVPacket pkt;

    av_init_packet(&pkt);
    for (int i = 0; i < 10000; i++)
    {
        int n = abcdk_ffeditor_read_packet(r, &pkt, -1);
        if (n < 0)
            break;

     //   abcdk_trace_printf(LOG_DEBUG,"src: DTS(%lld),PTS(%lld),DUR(%lld),",pkt.dts,pkt.pts,pkt.duration*1000);

        abcdk_rtsp_server_play_stream(ctx, media, stream[pkt.stream_index], pkt.data , pkt.size , pkt.duration/abcdk_ffeditor_fps(r,pkt.stream_index)*1000);
    }

    av_packet_unref(&pkt);

    abcdk_rtsp_server_remove_media(ctx,media);

END:


    abcdk_ffeditor_destroy(&r);

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