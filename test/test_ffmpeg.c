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
    const char *dst_fmt = abcdk_option_get(args,"--dst-fmt",0,"");

    abcdk_ffmpeg_t *r = abcdk_ffmpeg_open_capture(NULL,src,0);
    abcdk_ffmpeg_t *w = abcdk_ffmpeg_open_writer(dst_fmt,dst,NULL,5);

    AVFormatContext *rf = abcdk_ffmpeg_ctxptr(r);
    AVFormatContext *wf = abcdk_ffmpeg_ctxptr(w);

    abcdk_avformat_dump(rf,0);
    
    for(int i = 0;i<abcdk_ffmpeg_streams(r);i++)
    {
        AVStream * p = abcdk_ffmpeg_streamptr(r,i);

        // abcdk_hevc_extradata_t extradata = {0};

        // abcdk_hevc_extradata_deserialize(p->codec->extradata, p->codec->extradata_size, &extradata);
       
        AVCodecContext *opt = abcdk_avcodec_alloc3(p->codec->codec_id,1);

        abcdk_avstream_parameters_to_context(opt,p);

        opt->codec_tag = 0;
        //int fps = abcdk_avstream_fps(rf,p);
        //abcdk_avcodec_encode_video_fill_time_base(opt, 5);

        int n = abcdk_ffmpeg_add_stream(w,opt,1);

        wf->streams[n]->avg_frame_rate = p->avg_frame_rate;
        wf->streams[n]->r_frame_rate = p->r_frame_rate;

        abcdk_avcodec_free(&opt);
    }
    
    abcdk_ffmpeg_write_header(w,1);

    abcdk_avformat_dump(wf,1);

    AVPacket pkt;

    av_init_packet(&pkt);
    for(int i = 0;i<10000000;i++)
    {
        

        int n= abcdk_ffmpeg_read(r,&pkt,-1);
        if(n<0)
            break;

         abcdk_ffmpeg_write(w,&pkt,&rf->streams[n]->time_base);

     //   abcdk_ffmpeg_write2(w,pkt.data,pkt.size,0,n);
    }

    av_packet_unref(&pkt);


    abcdk_ffmpeg_write_trailer(w);
    abcdk_ffmpeg_destroy(&w);
    abcdk_ffmpeg_destroy(&r);
}

int abcdk_test_codec(abcdk_option_t *args)
{
    const char *src = abcdk_option_get(args,"--src",0,"");
    double src_xspeed = abcdk_option_get_double(args,"--src-xpeed",0,1);
    const char *dst = abcdk_option_get(args,"--dst",0,"");
    const char *dst_fmt = abcdk_option_get(args,"--dst-fmt",0,"");

    abcdk_ffmpeg_t *r = abcdk_ffmpeg_open_capture(NULL,src,30);
    abcdk_ffmpeg_t *w = abcdk_ffmpeg_open_writer(dst_fmt,dst,NULL,5);

    AVFormatContext *rf = abcdk_ffmpeg_ctxptr(r);
    AVFormatContext *wf = abcdk_ffmpeg_ctxptr(w);

    abcdk_avformat_dump(rf,0);
    
    for(int i = 0;i<abcdk_ffmpeg_streams(r);i++)
    {
        AVStream * p = abcdk_ffmpeg_streamptr(r,i);

        AVCodecContext *opt = abcdk_avcodec_alloc3(p->codec->codec_id,1);

        abcdk_avstream_parameters_to_context(opt,p);

        opt->codec_tag = 0;
        opt->gop_size = 12;
        int fps = abcdk_ffmpeg_fps(r,i,src_xspeed);
        abcdk_avcodec_encode_video_fill_time_base(opt, fps);

        abcdk_ffmpeg_add_stream(w,opt,0);

        abcdk_avcodec_free(&opt);
    }
    
    abcdk_ffmpeg_write_header(w,0);

    abcdk_avformat_dump(wf,1);

    AVFrame *inframe = av_frame_alloc();
    for(int i = 0;i<1000;i++)
    {
        abcdk_ffmpeg_read_delay(r,src_xspeed);
        
        int n= abcdk_ffmpeg_read2(r,inframe,-1);
        if(n<0)
            break;

         abcdk_ffmpeg_write3(w,inframe,n);
    }

    av_frame_free(&inframe);


    abcdk_ffmpeg_write_trailer(w);
    abcdk_ffmpeg_destroy(&w);
    abcdk_ffmpeg_destroy(&r);
}

int abcdk_test_extradata(abcdk_option_t *args)
{
    const char *src = abcdk_option_get(args,"--src",0,"");
    const char *dst = abcdk_option_get(args,"--dst",0,"");
    const char *dst_fmt = abcdk_option_get(args,"--dst-fmt",0,"");

    abcdk_ffmpeg_t *r = abcdk_ffmpeg_open_capture(NULL,src,0);
    abcdk_ffmpeg_t *w = abcdk_ffmpeg_open_writer(dst_fmt,dst,NULL,5);

    AVStream *vs_p = abcdk_ffmpeg_streamptr(r,0);

    abcdk_hexdump(stderr,vs_p->codec->extradata,vs_p->codec->extradata_size,0,NULL);

    if (vs_p->codec->codec_id == AV_CODEC_ID_H264)
    {
        abcdk_h264_extradata_t extradata = {0};

        abcdk_h264_extradata_deserialize(vs_p->codec->extradata, vs_p->codec->extradata_size, &extradata);

        abcdk_hexdump(stderr, extradata.sps->pptrs[0], extradata.sps->sizes[0], 0, NULL);
        abcdk_hexdump(stderr, extradata.pps->pptrs[0], extradata.pps->sizes[0], 0, NULL);
    }
    else if (vs_p->codec->codec_id == AV_CODEC_ID_HEVC)
    {
        abcdk_hevc_extradata_t extradata = {0};

        abcdk_hevc_extradata_deserialize(vs_p->codec->extradata, vs_p->codec->extradata_size, &extradata);

        for (int i = 0; i < extradata.nal_array_num; i++)
        {
            struct _nal_array *nal_p = &extradata.nal_array[i];

            for (int j = 0; j < nal_p->nal_num; j++)
                abcdk_hexdump(stderr, nal_p->nal->pptrs[j], nal_p->nal->sizes[j], 0, NULL);
        }
    }

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
    else if(cmd == 2)
        abcdk_test_codec(args);
    else if(cmd == 3)
        abcdk_test_extradata(args);

#endif //HAVE_FFMPEG

    return 0;
}