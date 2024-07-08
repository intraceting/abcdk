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

static abcdk_ffserver_t *ffserver_ctx = NULL;


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

static void stream_destructor_cb(void *opaque, abcdk_object_t *stream)
{
  //  node_t *p = (node_t*)abcdk_https_get_userdata(stream);

    
}



static void stream_construct_cb(void *opaque, abcdk_object_t *stream)
{

}

static void stream_request_cb(void *opaque, abcdk_object_t *stream)
{
    abcdk_https_response_header_set(stream, "Status","%d",200);
    abcdk_https_response_header_set(stream, "Content-Type","%s","video/mp4");
//    abcdk_https_response_header_set(stream, "Content-Length","1234567890");

    abcdk_https_response_header_end(stream);

}

static void stream_output_cb(void *opaque, abcdk_object_t *stream)
{
   //  node_t *p = (node_t*)abcdk_https_get_userdata(stream);

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

    /*等待终止信号。*/
    abcdk_proc_wait_exit_signal(-1);

final:

    abcdk_https_destroy(&io_ctx);
    abcdk_https_session_unref(&listen_p);
}

#else

int abcdk_test_fmp4(abcdk_option_t *args)
{
    return 0;
}

#endif // HAVE_FFMPEG
