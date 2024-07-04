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

typedef struct _node
{
    abcdk_stream_t *send_buf;

    abcdk_ffmpeg_t *reader;
    abcdk_ffmpeg_t *writer;

    abcdk_ffmpeg_config_t rcfg;
    abcdk_ffmpeg_config_t wcfg;
    
}node_t;

static void session_prepare_cb(void *opaque, abcdk_httpd_session_t **session, abcdk_httpd_session_t *listen)
{
    abcdk_httpd_session_t *p;

    p = abcdk_httpd_session_alloc((abcdk_httpd_t*)opaque);


    *session = p;
}

static void session_accept_cb(void *opaque, abcdk_httpd_session_t *session, int *result)
{
    *result = 0;
}

static void session_ready_cb(void *opaque, abcdk_httpd_session_t *session)
{

}

static void session_close_cb(void *opaque, abcdk_httpd_session_t *session)
{

}

static void stream_destructor_cb(void *opaque, abcdk_object_t *stream)
{
    node_t *p = (node_t*)abcdk_httpd_get_userdata(stream);

    abcdk_stream_destroy(&p->send_buf);
    abcdk_ffmpeg_destroy(&p->reader);
    abcdk_ffmpeg_destroy(&p->writer);
    
}


int _ffmpeg_write_packet_cb(void *opaque, uint8_t *buf, int buf_size)
{
    abcdk_stream_t *send_buf_p = (abcdk_stream_t*)opaque;

    abcdk_stream_write_buffer(send_buf_p,buf,buf_size);

    return buf_size;
}


static void stream_construct_cb(void *opaque, abcdk_object_t *stream)
{
    node_t *p = abcdk_heap_alloc(sizeof(node_t));

    abcdk_httpd_set_userdata(stream,p);
}

static void stream_request_cb(void *opaque, abcdk_object_t *stream)
{
    node_t *p = (node_t*)abcdk_httpd_get_userdata(stream);

    p->rcfg.file_name = "rtsp://192.168.100.96/live/bbbb";
    p->rcfg.bit_stream_filter = 1;
    p->rcfg.timeout = 10;
    p->rcfg.read_speed = 1.0;
    p->rcfg.read_delay_max = 1.0;    
    p->reader = abcdk_ffmpeg_open(&p->rcfg);

    p->send_buf = abcdk_stream_create();

    p->wcfg.writer = 1;
    p->wcfg.io.opaque = p->send_buf;
    p->wcfg.io.write_cb = _ffmpeg_write_packet_cb;
    p->wcfg.short_name = "mp4";

    p->writer = abcdk_ffmpeg_open(&p->wcfg);

    AVFormatContext *rf = abcdk_ffmpeg_ctxptr(p->reader);
    AVFormatContext *wf = abcdk_ffmpeg_ctxptr(p->writer);

    for(int i = 0;i<abcdk_ffmpeg_streams(p->reader);i++)
    {
        AVStream * vs_p = abcdk_ffmpeg_streamptr(p->reader,i);

        AVCodecContext *opt = abcdk_avcodec_alloc3(vs_p->codecpar->codec_id,1);
        abcdk_avstream_parameters_to_context(opt,vs_p);

        opt->codec_tag = 0;
        //int fps = abcdk_avstream_fps(rf,p);
        //abcdk_avcodec_encode_video_fill_time_base(opt, 5);

        int n = abcdk_ffmpeg_add_stream(p->writer,opt,1);

        wf->streams[n]->avg_frame_rate = vs_p->avg_frame_rate;
        wf->streams[n]->r_frame_rate = vs_p->r_frame_rate;

        abcdk_avcodec_free(&opt);
    }

    abcdk_ffmpeg_write_header(p->writer,1);

    abcdk_avformat_dump(wf,1);

    abcdk_httpd_response_header_set(stream, "Status","%d",200);
    abcdk_httpd_response_header_set(stream, "Content-Type","%s","video/mp4");
//    abcdk_httpd_response_header_set(stream, "Content-Length","1234567890");

    abcdk_httpd_response_header_end(stream);

    // abcdk_object_t *buf = abcdk_object_alloc2(128*1024);
    // int rlen = abcdk_stream_read(p->send_buf,buf->pptrs[0],buf->sizes[0]);
    // if(rlen >0)
    // {
    //     buf->sizes[0] = rlen;
    //     abcdk_httpd_response(stream,buf);
    // }
    // else
    // {
    //     abcdk_httpd_response(stream,NULL);
    // }
}

static void stream_output_cb(void *opaque, abcdk_object_t *stream)
{
     node_t *p = (node_t*)abcdk_httpd_get_userdata(stream);
     int eof = 0;
     abcdk_object_t *buf ;

TRY:

    buf = abcdk_object_alloc2(128*1024);
    int rlen = abcdk_stream_read(p->send_buf,buf->pptrs[0],buf->sizes[0]);
    if(rlen >0)
    {
        buf->sizes[0] = rlen;
        abcdk_httpd_response(stream,buf);

        fprintf(stderr,"rlen(%lld)\n",rlen);
    }
    else 
    {

        abcdk_object_unref(&buf);

        if(eof)
        {
            abcdk_httpd_response(stream,NULL);
            return ;
        }

        AVFormatContext *rf = abcdk_ffmpeg_ctxptr(p->reader);

        AVPacket pkt;
        av_init_packet(&pkt);


        abcdk_ffmpeg_read_delay(p->reader);

        int n= abcdk_ffmpeg_read_packet(p->reader,&pkt,-1);
        if(n<0)
        {
            abcdk_ffmpeg_write_trailer(p->writer);
            eof = 1;
        }
        else 
        {
            fprintf(stderr,"pts(%lld),dts(%lld)\n",pkt.pts,pkt.dts);

            abcdk_ffmpeg_write_packet(p->writer, &pkt, &rf->streams[n]->time_base);
        }

        

        av_packet_unref(&pkt);

        goto TRY;
    }
}

int abcdk_test_fmp4(abcdk_option_t *args)
{
    abcdk_sockaddr_t listen_addr = {0};
    abcdk_httpd_config_t cfg = {0};
    abcdk_httpd_t *io_ctx;
    abcdk_httpd_session_t *listen_p;

    abcdk_sockaddr_from_string(&listen_addr, "0.0.0.0:1111", 0);

    io_ctx = abcdk_httpd_create(123, -1);

    cfg.opaque = io_ctx;
    cfg.session_prepare_cb = session_prepare_cb;
    cfg.session_accept_cb = session_accept_cb;
    cfg.session_ready_cb = session_ready_cb;
    cfg.session_close_cb = session_close_cb;
    cfg.stream_destructor_cb = stream_destructor_cb;
    cfg.stream_construct_cb = stream_construct_cb;
    cfg.stream_request_cb = stream_request_cb;
    cfg.stream_output_cb = stream_output_cb;
    cfg.req_max_size = 123456789;
    cfg.req_tmp_path = NULL;
    cfg.name = "test_fmp4";
    cfg.realm = "httpd";
    cfg.auth_path = NULL;
    cfg.a_c_a_o = "*";

    listen_p = abcdk_httpd_session_alloc(io_ctx);

    abcdk_httpd_session_listen(listen_p,&listen_addr,&cfg);

    /*等待终止信号。*/
    abcdk_proc_wait_exit_signal(-1);

final:

    abcdk_httpd_destroy(&io_ctx);
    abcdk_httpd_session_unref(&listen_p);
}

#else

int abcdk_test_fmp4(abcdk_option_t *args)
{
    return 0;
}

#endif // HAVE_FFMPEG
