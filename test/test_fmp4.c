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
    abcdk_ffserver_config_t live_cfg;
    abcdk_ffserver_task_t *task_ctx;
    abcdk_stream_t *live_buf;
}node_t;

static abcdk_ffserver_t *g_ffserver_ctx = NULL;


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

    abcdk_stream_destroy(&p->live_buf);
    abcdk_heap_free(p);
}


static void stream_construct_cb(void *opaque, abcdk_https_stream_t *stream)
{
    node_t * p = abcdk_heap_alloc(sizeof(node_t));

    abcdk_https_set_userdata(stream,p);
}

static void stream_close_cb(void *opaque,abcdk_https_stream_t *stream)
{
    node_t *p = (node_t*)abcdk_https_get_userdata(stream);

    abcdk_ffserver_task_del(g_ffserver_ctx,&p->task_ctx);
}

static void _live_delete_cb(void *opaque)
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

    p->live_buf = abcdk_stream_create();
    p->live_cfg.flag = 3;
    p->live_cfg.u.live.buf = p->live_buf;
    p->live_cfg.u.live.delay_max = 3.0;
    p->live_cfg.u.live.ready_cb = _live_ready_cb;
    p->live_cfg.u.live.delete_cb = _live_delete_cb;
    p->live_cfg.u.live.opaque = abcdk_https_refer(stream);

    p->task_ctx = abcdk_ffserver_task_add(g_ffserver_ctx,&p->live_cfg);

    abcdk_https_response_header_set(stream, "Status","%d",200);
    abcdk_https_response_header_set(stream, "Content-Type","%s","video/mp4");
//    abcdk_https_response_header_set(stream, "Content-Length","1234567890");

    abcdk_https_response_header_end(stream);

}

static void stream_output_cb(void *opaque, abcdk_https_stream_t *stream)
{
    node_t *p = (node_t *)abcdk_https_get_userdata(stream);

    abcdk_object_t *buf = abcdk_object_alloc2(1000000);

    int rlen = abcdk_stream_read(p->live_buf, buf->pptrs[0], buf->sizes[0]);
    if (rlen > 0)
    {
        buf->sizes[0] = rlen;
        abcdk_https_response(stream, buf);
    }
    else
    {
        abcdk_object_unref(&buf);
    }

    abcdk_ffserver_task_heartbeat(g_ffserver_ctx,p->task_ctx);
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
    cfg.name = "test_fmp4";
    cfg.realm = "httpd";
    cfg.auth_path = NULL;
    cfg.a_c_a_o = "*";

    listen_p = abcdk_https_session_alloc(io_ctx);

    abcdk_https_session_listen(listen_p,&listen_addr,&cfg);

    abcdk_ffserver_config_t src_cfg ={0};
    abcdk_ffserver_config_t record_cfg ={0};
    abcdk_ffserver_config_t push_cfg ={0};

    src_cfg.u.src.url = abcdk_option_get(args,"--src",0,"");
    src_cfg.u.src.speed = 1.0;
    src_cfg.u.src.delay_max = 3.0;
    src_cfg.u.src.timeout = 5.0;



    g_ffserver_ctx = abcdk_ffserver_create(&src_cfg);

    record_cfg.flag = 1;
    record_cfg.u.record.prefix = "/tmp/cccc/cccc_";
    record_cfg.u.record.count = 10;
    record_cfg.u.record.duration = 5;

    push_cfg.flag = 2;
    push_cfg.u.push.url = "rtmp://192.168.100.96/live/cccc";
    push_cfg.u.push.fmt = "rtmp";

    abcdk_ffserver_task_add(g_ffserver_ctx,&record_cfg);
    abcdk_ffserver_task_add(g_ffserver_ctx,&push_cfg);

    /*等待终止信号。*/
    abcdk_proc_wait_exit_signal(-1);

final:

    abcdk_ffserver_destroy(&g_ffserver_ctx);

    abcdk_https_destroy(&io_ctx);
    abcdk_https_session_unref(&listen_p);
}

#else

int abcdk_test_fmp4(abcdk_option_t *args)
{
    return 0;
}

#endif // HAVE_FFMPEG
