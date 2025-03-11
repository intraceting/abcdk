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
    abcdk_ffeditor_config_t rcfg = {0},wcfg = {1};

    rcfg.file_name = abcdk_option_get(args,"--src",0,"");
    rcfg.read_speed = abcdk_option_get_double(args,"--src-xpeed",0,1);
    rcfg.read_delay_max = abcdk_option_get_double(args,"--src-delay-max",0,1);
    rcfg.bit_stream_filter = 1;
    wcfg.file_name = abcdk_option_get(args,"--dst",0,"");
    wcfg.short_name = abcdk_option_get(args,"--dst-fmt",0,"");

    abcdk_ffeditor_t *r = abcdk_ffeditor_open(&rcfg);
    abcdk_ffeditor_t *w = abcdk_ffeditor_open(&wcfg);

    AVFormatContext *rf = abcdk_ffeditor_ctxptr(r);
    AVFormatContext *wf = abcdk_ffeditor_ctxptr(w);

    abcdk_avformat_dump(rf,0);
    
    for(int i = 0;i<abcdk_ffeditor_streams(r);i++)
    {
        AVStream * p = abcdk_ffeditor_streamptr(r,i);

        if(p->codecpar->codec_id == AV_CODEC_ID_HEVC)
        {   
            abcdk_hevc_extradata_t extradata = {0};
            abcdk_hevc_extradata_deserialize(p->codecpar->extradata, p->codecpar->extradata_size, &extradata);
        }
        else if(p->codecpar->codec_id == AV_CODEC_ID_H264)
        {
            abcdk_h264_extradata_t extradata = {0};
            abcdk_h264_extradata_deserialize(p->codecpar->extradata, p->codecpar->extradata_size, &extradata);
        }
       
        AVCodecContext *opt = abcdk_avcodec_alloc3(p->codecpar->codec_id,1);

        abcdk_avstream_parameters_to_context(opt,p);

        opt->codec_tag = 0;
        // int fps = abcdk_ffeditor_fps(r,p->index);
        // abcdk_avcodec_encode_video_fill_time_base(opt, fps);

        int n = abcdk_ffeditor_add_stream(w,opt,1);

        wf->streams[n]->avg_frame_rate = p->avg_frame_rate;
        wf->streams[n]->r_frame_rate = p->r_frame_rate;

        abcdk_avcodec_free(&opt);
    }

    AVDictionary *dict = NULL;

   /// av_dict_set(&dict, "hls_segment_filename", "/tmp/ccc/aaaa%d.ts", 0);
   // av_dict_set(&dict, "hls_time", "2", 0);
  //  av_dict_set(&dict, "hls_list_size","5", 0);
  //  av_dict_set(&dict, "start_number", "1", 0);
  //  av_dict_set(&dict, "hls_delete_threshold","100",0);
  //  av_dict_set(&dict, "hls_flags","+delete_segments+append_list+omit_endlist+temp_file",0);

   //   av_dict_set(&dict, "hls_flags","append_list+omit_endlist+temp_file",0);
    
    av_dict_set(&dict, "movflags", "empty_moov+default_base_moof+frag_keyframe", 0);
        
    abcdk_ffeditor_write_header0(w,dict);

    av_dict_free(&dict);

    abcdk_avformat_dump(wf,1);

    uint64_t pos[2] = {1,0};

    AVPacket pkt;

    av_init_packet(&pkt);
    for(int i = 0;i<1000;i++)
    {
       // abcdk_ffeditor_read_delay(r,0);

        int n= abcdk_ffeditor_read_packet(r,&pkt,-1);
        if(n<0)
            break;

        int m =0;
        if(rf->streams[n]->codecpar->codec_id == AV_CODEC_ID_H264)
            m = abcdk_h264_idr(pkt.data,pkt.size);
        else if(rf->streams[n]->codecpar->codec_id == AV_CODEC_ID_HEVC)
            m = abcdk_hevc_irap(pkt.data,pkt.size);

        double psec = abcdk_ffeditor_ts2sec(r, pkt.stream_index, pkt.pts);
        double dsec = abcdk_ffeditor_ts2sec(r, pkt.stream_index, pkt.dts);

        double dru = (double)pkt.duration * abcdk_avmatch_r2d(rf->streams[n]->time_base,rcfg.read_speed);

        fprintf(stderr, "flag(%d,%d),pts(%.6f),dts(%.6f),dur(%.6f)\n",m, pkt.flags, psec, dsec, dru);

        abcdk_ffeditor_write_packet(w, &pkt, &rf->streams[n]->time_base);

        //   abcdk_ffeditor_write_packet(w,pkt.data,pkt.size,0,n);

      //  abcdk_file_segment(NULL,"/tmp/ccc/aaaa%llu.ts",10,1,pos);
    }

    av_packet_unref(&pkt);


    abcdk_ffeditor_write_trailer(w);
    abcdk_ffeditor_destroy(&w);
    abcdk_ffeditor_destroy(&r);
}

int abcdk_test_codec(abcdk_option_t *args)
{
    abcdk_ffeditor_config_t rcfg = {0},wcfg = {1};

    rcfg.file_name = abcdk_option_get(args,"--src",0,"");
    rcfg.read_speed = abcdk_option_get_double(args,"--src-xpeed",0,1);
    rcfg.read_delay_max = abcdk_option_get_double(args,"--src-delay-max",0,1);
    rcfg.bit_stream_filter = 1;
    wcfg.file_name = abcdk_option_get(args,"--dst",0,"");
    wcfg.short_name = abcdk_option_get(args,"--dst-fmt",0,"");

    abcdk_ffeditor_t *r = abcdk_ffeditor_open(&rcfg);
    abcdk_ffeditor_t *w = abcdk_ffeditor_open(&wcfg);

    AVFormatContext *rf = abcdk_ffeditor_ctxptr(r);
    AVFormatContext *wf = abcdk_ffeditor_ctxptr(w);

    abcdk_avformat_dump(rf,0);
    
    for(int i = 0;i<abcdk_ffeditor_streams(r);i++)
    {
        AVStream * p = abcdk_ffeditor_streamptr(r,i);

        AVCodecContext *opt = abcdk_avcodec_alloc3(p->codecpar->codec_id,1);

        abcdk_avstream_parameters_to_context(opt,p);

        int fps = abcdk_ffeditor_fps(r,i);

        opt->codec_tag = 0;
        opt->gop_size = fps;
        
        abcdk_avcodec_encode_video_fill_time_base(opt, fps);

        abcdk_ffeditor_add_stream(w,opt,0);

        abcdk_avcodec_free(&opt);
    }
    
    abcdk_ffeditor_write_header(w,0);

    abcdk_avformat_dump(wf,1);

    AVFrame *inframe = av_frame_alloc();
    for(int i = 0;i<10000;i++)
    {
       // abcdk_ffeditor_read_delay(r,0);
        
        int n= abcdk_ffeditor_read_frame(r,inframe,-1);
        if(n<0)
            break;

         abcdk_ffeditor_write_frame(w,inframe,n);

         usleep(rand() %40 * 1000);
    }

    av_frame_free(&inframe);


    abcdk_ffeditor_write_trailer(w);
    abcdk_ffeditor_destroy(&w);
    abcdk_ffeditor_destroy(&r);
}

int abcdk_test_extradata(abcdk_option_t *args)
{
    abcdk_ffeditor_config_t rcfg = {0},wcfg = {1};

    rcfg.file_name = abcdk_option_get(args,"--src",0,"");
    rcfg.read_speed = abcdk_option_get_double(args,"--src-xpeed",0,1);
    rcfg.bit_stream_filter = 1;
    wcfg.file_name = abcdk_option_get(args,"--dst",0,"");
    wcfg.short_name = abcdk_option_get(args,"--dst-fmt",0,"");

    abcdk_ffeditor_t *r = abcdk_ffeditor_open(&rcfg);
    abcdk_ffeditor_t *w = abcdk_ffeditor_open(&wcfg);

    AVStream *vs_p = abcdk_ffeditor_streamptr(r,0);

    abcdk_hexdump(stderr,vs_p->codecpar->extradata,vs_p->codecpar->extradata_size,0,NULL);

    if (vs_p->codecpar->codec_id == AV_CODEC_ID_H264)
    {
        abcdk_h264_extradata_t extradata = {0};

        abcdk_h264_extradata_deserialize(vs_p->codecpar->extradata, vs_p->codecpar->extradata_size, &extradata);

        abcdk_hexdump(stderr, extradata.sps->pptrs[0], extradata.sps->sizes[0], 0, NULL);
        abcdk_hexdump(stderr, extradata.pps->pptrs[0], extradata.pps->sizes[0], 0, NULL);
    }
    else if (vs_p->codecpar->codec_id == AV_CODEC_ID_HEVC)
    {
        abcdk_hevc_extradata_t extradata = {0};

        abcdk_hevc_extradata_deserialize(vs_p->codecpar->extradata, vs_p->codecpar->extradata_size, &extradata);

        for (int i = 0; i < extradata.nal_array_num; i++)
        {
            struct _nal_array *nal_p = &extradata.nal_array[i];

            for (int j = 0; j < nal_p->nal_num; j++)
                abcdk_hexdump(stderr, nal_p->nal->pptrs[j], nal_p->nal->sizes[j], 0, NULL);
        }
    }

    abcdk_ffeditor_destroy(&w);
    abcdk_ffeditor_destroy(&r);
}

int abcdk_test_audio(abcdk_option_t *args)
{
    abcdk_ffeditor_config_t rcfg = {0};

    rcfg.file_name = abcdk_option_get(args,"--src",0,"");
    rcfg.read_speed = abcdk_option_get_double(args,"--src-xpeed",0,1);
    rcfg.bit_stream_filter = 1;

    abcdk_ffeditor_t *r = abcdk_ffeditor_open(&rcfg);

    AVStream *vs_p = abcdk_ffeditor_find_stream(r,AVMEDIA_TYPE_AUDIO);

    int stream_idx = vs_p->index;

    AVFrame *inframe = av_frame_alloc();
    for(int i = 0;i<1000;i++)
    {
       // abcdk_ffeditor_read_delay(r,0);
        
        int n = abcdk_ffeditor_read_frame(r,inframe,stream_idx);
        if(n<0)
            break;

         abcdk_hexdump(stderr, inframe->data[0],inframe->linesize[0], 0, NULL);
    }

    av_frame_free(&inframe);


    abcdk_ffeditor_destroy(&r);
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

#endif //HAVE_FFMPEG

    return 0;
}
