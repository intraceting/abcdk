/*
 * This file is part of ABCDK.
 *
 * MIT License
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
    const char *src = abcdk_option_get(args,"--src",0,"");
    const char *dst = abcdk_option_get(args,"--dst",0,"");

    abcdk_ffmpeg_t *r = abcdk_ffmpeg_open_capture(NULL,src,NULL,0);
    abcdk_ffmpeg_t *w = abcdk_ffmpeg_open_writer(NULL,dst,NULL,NULL);

    AVFormatContext *rf = abcdk_ffmpeg_ctxptr(r);
    
    abcdk_avcodec_parameters_t rparam[10] = {0};
    for(int i = 0;i<rf->nb_streams;i++)
    {
        AVStream * p = rf->streams[i];
        abcdk_avcodec_parameters_t *p2 = &rparam[i];

        p2->codec_id = p->codec->codec_id;
        p2->extradata = p->codec->extradata;
        p2->extradata_size = p->codec->extradata_size;
        p2->bit_rate = p->codec->bit_rate;

        if (p->codec->codec_type == AVMEDIA_TYPE_VIDEO)
        {
            p2->fps = abcdk_avstream_fps(rf, p);
            p2->width = abcdk_avstream_width(rf, p);
            p2->height = abcdk_avstream_height(rf, p);
        }
        else if (p->codec->codec_type == AVMEDIA_TYPE_AUDIO)
        {
            p2->fps = p->codec->sample_rate;
            p2->channels = p->codec->channels;
            p2->sample_rate = p->codec->sample_rate;
            p2->channel_layout = p->codec->channel_layout;
            p2->frame_size = p->codec->frame_size;
        }
        else
        {
            continue;
        }

        abcdk_ffmpeg_add_stream(w,p2,1);
    }
    
    abcdk_ffmpeg_write_header(w,0);

    AVPacket pkt;

    av_init_packet(&pkt);
    while(1)
    {
        int n= abcdk_ffmpeg_read(r,&pkt,-1);
        if(n<0)
            break;

    //    pkt.pos = -1;
     //   abcdk_ffmpeg_write(w,&pkt);

        abcdk_ffmpeg_write2(w,pkt.data,pkt.size,0,n);
    }

    av_packet_unref(&pkt);


    abcdk_ffmpeg_write_trailer(w);
    abcdk_ffmpeg_destroy(&w);
    abcdk_ffmpeg_destroy(&r);
}
#endif //HAVE_FFMPEG
int abcdk_test_ffmpeg(abcdk_option_t *args)
{
#ifdef HAVE_FFMPEG

    int cmd = abcdk_option_get_int(args,"--cmd",0,1);

    if(cmd == 1)
        abcdk_test_record(args);

#endif //HAVE_FFMPEG

    return 0;
}