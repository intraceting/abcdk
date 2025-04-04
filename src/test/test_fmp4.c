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

typedef struct _node
{
    /**流缓存。*/
    abcdk_stream_t *buf;

    abcdk_ffmpeg_nvr_config_t live_cfg;
    char task_id[33];
} node_t;

static abcdk_ffmpeg_nvr_t *g_ff_nvr_ctx = NULL;


static void session_prepare_cb(void *opaque, abcdk_https_session_t **session, abcdk_https_session_t *listen)
{
    abcdk_https_session_t *p;

    p = abcdk_https_session_alloc((abcdk_https_t*)opaque);


    *session = p;
}

static void session_accept_cb(void *opaque, abcdk_https_session_t *session, int *result)
{
    *result = 0;
}

static void session_ready_cb(void *opaque, abcdk_https_session_t *session)
{

}

static void session_close_cb(void *opaque, abcdk_https_session_t *session)
{

}

static void stream_destructor_cb(void *opaque, abcdk_https_stream_t *stream)
{
    node_t *p = (node_t*)abcdk_https_get_userdata(stream);

    abcdk_stream_destroy(&p->buf);
    abcdk_heap_free(p);
}


static void stream_construct_cb(void *opaque, abcdk_https_stream_t *stream)
{
    node_t * p = abcdk_heap_alloc(sizeof(node_t));

    abcdk_https_set_userdata(stream,p);

    p->buf = abcdk_stream_create();
}

static void stream_close_cb(void *opaque,abcdk_https_stream_t *stream)
{
    node_t *p = (node_t*)abcdk_https_get_userdata(stream);

    abcdk_ffmpeg_nvr_task_del(g_ff_nvr_ctx,p->task_id);
}

static void _live_remove_cb(void *opaque)
{
    abcdk_https_stream_t *stream_p =(abcdk_https_stream_t *)opaque;

    abcdk_https_unref(&stream_p);
}

static void _live_ready_cb(void *opaque)
{
    abcdk_https_stream_t *stream_p = (abcdk_https_stream_t*)opaque;

    abcdk_https_response_ready(stream_p);
}

static void stream_request_cb(void *opaque, abcdk_https_stream_t *stream)
{
    node_t *p = (node_t*)abcdk_https_get_userdata(stream);

    p->live_cfg.opaque = abcdk_https_refer(stream);
    p->live_cfg.flag = ABCDK_FFMPEG_NVR_CFG_FLAG_LIVE;
    p->live_cfg.u.live.ready_cb = _live_ready_cb;
    p->live_cfg.u.live.delay_max = 3.0;
    p->live_cfg.u.live.buf = p->buf;

    abcdk_rand_bytes(p->task_id,32,1);

    abcdk_ffmpeg_nvr_task_add(g_ff_nvr_ctx,p->task_id, &p->live_cfg);

    abcdk_https_response_header_set(stream, "Status","%d",200);
    abcdk_https_response_header_set(stream, "Content-Type","%s","video/mp4");
//    abcdk_https_response_header_set(stream, "Content-Length","1234567890");

    abcdk_https_response_header_end(stream);

}

static void stream_output_cb(void *opaque, abcdk_https_stream_t *stream)
{
    node_t *p = (node_t *)abcdk_https_get_userdata(stream);

    abcdk_object_t *data = abcdk_object_alloc2(10000);

    int chk = abcdk_stream_read(p->buf, data->pptrs[0], data->sizes[0]);
    if (chk > 0)
    {
        data->sizes[0] = chk;

        chk = abcdk_https_response(stream, data);
        if (chk != 0)
            abcdk_object_unref(&data);
    }
    else
    {
        abcdk_object_unref(&data);
    }

    abcdk_ffmpeg_nvr_task_heartbeat(g_ff_nvr_ctx, p->task_id);
}

int abcdk_test_fmp4(abcdk_option_t *args)
{
    abcdk_sockaddr_t listen_addr = {0};
    abcdk_https_config_t cfg = {0};
    abcdk_https_t *io_ctx;
    abcdk_https_session_t *listen_p;

    abcdk_sockaddr_from_string(&listen_addr, "0.0.0.0:1111", 0);

    io_ctx = abcdk_https_create(123, -1);

    cfg.opaque = io_ctx;
    cfg.session_prepare_cb = session_prepare_cb;
    cfg.session_accept_cb = session_accept_cb;
    cfg.session_ready_cb = session_ready_cb;
    cfg.session_close_cb = session_close_cb;
    cfg.stream_destructor_cb = stream_destructor_cb;
    cfg.stream_construct_cb = stream_construct_cb;
    cfg.stream_close_cb = stream_close_cb;
    cfg.stream_request_cb = stream_request_cb;
    cfg.stream_output_cb = stream_output_cb;
    cfg.req_max_size = 123456789;
    cfg.req_tmp_path = NULL;
    cfg.name = "test_fmp4.mp4";
    cfg.realm = "httpd";
    cfg.auth_path = NULL;
    cfg.a_c_a_o = "*";

    listen_p = abcdk_https_session_alloc(io_ctx);

    abcdk_https_session_listen(listen_p,&listen_addr,&cfg);

    abcdk_ffmpeg_nvr_config_t src_cfg ={0};
    abcdk_ffmpeg_nvr_config_t record_cfg ={0};
    abcdk_ffmpeg_nvr_config_t push_cfg ={0};

    src_cfg.u.src.url = abcdk_option_get(args,"--src",0,"");
    src_cfg.u.src.speed = 1.0;
    src_cfg.u.src.delay_max = 3.0;
    src_cfg.u.src.timeout = 5.0;



    g_ff_nvr_ctx = abcdk_ffmpeg_nvr_create(&src_cfg);

    record_cfg.flag = 1;
    record_cfg.u.record.prefix = "/tmp/cccc/cccc_";
    record_cfg.u.record.count = 10;
    record_cfg.u.record.duration = 60;

    char record_id[33] = {0};
    abcdk_rand_bytes(record_id,32,1);

    push_cfg.flag = 2;
  //  push_cfg.u.push.url = "rtmp://192.168.100.96/live/cccc";
  //  push_cfg.u.push.fmt = "rtmp";
    push_cfg.u.push.url = "rtsp://192.168.100.96/live/cccc";
    push_cfg.u.push.fmt = "rtsp";

    char push_id[33] = {0};
    abcdk_rand_bytes(push_id,32,1);

    abcdk_ffmpeg_nvr_task_add(g_ff_nvr_ctx,record_id,&record_cfg);
    abcdk_ffmpeg_nvr_task_add(g_ff_nvr_ctx,push_id,&push_cfg);

    /*等待终止信号。*/
    abcdk_proc_wait_exit_signal(-1);

final:

    abcdk_ffmpeg_nvr_destroy(&g_ff_nvr_ctx);

    abcdk_https_destroy(&io_ctx);
    abcdk_https_session_unref(&listen_p);
}

#else

int abcdk_test_fmp4(abcdk_option_t *args)
{
    return 0;
}

#endif // HAVE_FFMPEG
