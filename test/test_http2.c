/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"



#ifdef HAVE_NGHTTP2
#include <nghttp2/nghttp2.h>
#endif // HAVE_NGHTTP2

//#ifdef NGHTTP2_H
#if 0

typedef struct _h2_node
{
    abcdk_stcp_t *ctx;

    SSL_CTX *ssl_ctx;

    int protocol;

    nghttp2_session_callbacks *callbacks;
    nghttp2_session *session;
} h2_node_t;

typedef struct _h2_req_hdr{
    abcdk_object_t *method;
    abcdk_object_t *path;
    abcdk_object_t *scheme;
    abcdk_object_t *authority;
    abcdk_object_t *others[100];
    size_t num_others;
} h2_req_hdr_t;

#define ARRLEN(x) (sizeof(x) / sizeof(x[0]))

#define MAKE_NV(NAME, VALUE)                                                   \
  {                                                                            \
    (uint8_t *)NAME, (uint8_t *)VALUE, sizeof(NAME) - 1, sizeof(VALUE) - 1,    \
        NGHTTP2_NV_FLAG_NONE                                                   \
  }

ssize_t on_data_source_read_callback(
    nghttp2_session *session, int32_t stream_id, uint8_t *buf, size_t length,
    uint32_t *data_flags, nghttp2_data_source *source, void *user_data)
{
    abcdk_stcp_node_t *node = (abcdk_stcp_node_t *)user_data;

    size_t s = ABCDK_MIN(6LLU, length);
    strncpy((char *)buf, "ahaha\n", s);

    *data_flags |= NGHTTP2_DATA_FLAG_EOF;

    return s;
}

static void test_submit_response(nghttp2_session *session, int32_t stream_id)
{
    nghttp2_nv hdrs[] = {MAKE_NV(":status", "200"), MAKE_NV("content-type", "text/plain"), MAKE_NV("content-length", "6")};

    nghttp2_data_provider data_prd;
    data_prd.source.fd = -1;
    data_prd.read_callback = on_data_source_read_callback;

    int rv = nghttp2_submit_response(session, stream_id, hdrs, ABCDK_ARRAY_SIZE(hdrs), &data_prd);
    assert(rv == 0);
}

// nghttp2回调函数，用于处理HTTP/2会话事件
static int on_data_chunk_recv_callback(nghttp2_session *session, uint8_t flags, int32_t stream_id, const uint8_t *data, size_t len, void *user_data)
{
    // 在这里处理接收到的数据
    // 这里只是简单地打印接收到的数据
    printf("Received data on stream %d: %.*s\n", stream_id, (int)len, data);

    abcdk_stcp_node_t *node = (abcdk_stcp_node_t *)user_data;

    if (flags & NGHTTP2_FLAG_END_STREAM)
    {
        test_submit_response(session,stream_id);

        /*通知链路有数据要发送。*/
        abcdk_stcp_send_watch(node);
    }

    // 返回成功
    return 0;
}

// nghttp2回调函数，用于处理HTTP/2帧头部
static int on_header_callback(nghttp2_session *session, const nghttp2_frame *frame, const uint8_t *name, size_t namelen, const uint8_t *value, size_t valuelen, uint8_t flags, void *user_data)
{
    // 在这里处理帧头部
    // 这里只是简单地打印头部信息
    printf("Received header on stream %d: %.*s: %.*s\n", frame->hd.stream_id, (int)namelen, name, (int)valuelen, value);

    abcdk_stcp_node_t *node = (abcdk_stcp_node_t *)user_data;

    if(frame->hd.flags &NGHTTP2_FLAG_END_HEADERS)
    {
        test_submit_response(session,frame->hd.stream_id);

        /*通知链路有数据要发送。*/
        abcdk_stcp_send_watch(node);
    }

    // 返回成功
    return 0;
}

// nghttp2回调函数，用于处理流关闭事件
static int on_stream_close_callback(nghttp2_session *session, int32_t stream_id, uint32_t error_code, void *user_data)
{
    // 在这里处理流关闭事件
    printf("Stream %d closed with error code %u\n", stream_id, error_code);

    // 返回成功
    return 0;
}

// 发送数据的回调函数
static ssize_t on_send_callback(nghttp2_session *session, const uint8_t *data, size_t length, int flags, void *user_data) {
    // 在这里实现具体的发送逻辑，例如使用 socket 发送数据
    // 这里的示例代码仅为演示，实际情况需要根据你的应用程序做适当修改
    //printf("Sending data: %.*s\n", (int)length, data);

    abcdk_stcp_node_t *node = (abcdk_stcp_node_t *)user_data;

    abcdk_stcp_post_buffer(node,data,length);
    
    // 返回实际发送的字节数
    return length;
}

static void _node_destroy_cb(void *userdata)
{
    h2_node_t *http_p;

    http_p = (h2_node_t *)userdata;
    if (!http_p)
        return;

    if (http_p->session)
        nghttp2_session_del(http_p->session);
    if (http_p->callbacks)
        nghttp2_session_callbacks_del(http_p->callbacks);
}

static abcdk_stcp_node_t *_node_new(abcdk_stcp_t *ctx)
{
    h2_node_t *http_p = NULL;
    abcdk_stcp_node_t *node_new;

    node_new = abcdk_stcp_alloc(ctx, sizeof(h2_node_t),_node_destroy_cb);

    http_p = (h2_node_t *)abcdk_stcp_get_userdata(node_new);

    http_p->ctx = ctx;

    return node_new;
}

static void _prepare_cb(abcdk_stcp_node_t **node, abcdk_stcp_node_t *listen)
{
    h2_node_t *listen_p = NULL, *node_new_p = NULL;
    abcdk_stcp_node_t *node_new;

    listen_p = (h2_node_t *)abcdk_stcp_get_userdata(listen);

    node_new = _node_new(listen_p->ctx);
    node_new_p = (h2_node_t *)abcdk_stcp_get_userdata(node_new);

    if(listen_p->ssl_ctx)
        abcdk_stcp_upgrade2openssl(node_new,listen_p->ssl_ctx,1);

    // 初始化nghttp2回调结构
    nghttp2_session_callbacks_new(&node_new_p->callbacks);
    nghttp2_session_callbacks_set_on_data_chunk_recv_callback(node_new_p->callbacks, on_data_chunk_recv_callback);
    nghttp2_session_callbacks_set_on_header_callback(node_new_p->callbacks, on_header_callback);
    nghttp2_session_callbacks_set_on_stream_close_callback(node_new_p->callbacks, on_stream_close_callback);
    nghttp2_session_callbacks_set_send_callback(node_new_p->callbacks, on_send_callback);

    // 创建nghttp2服务器会话
    nghttp2_session_server_new(&node_new_p->session, node_new_p->callbacks, node_new);

    //nghttp2_session_set_user_data(node_new_p->session, node_new);

    nghttp2_settings_entry iv[1] = {
        {NGHTTP2_SETTINGS_MAX_CONCURRENT_STREAMS, 100}};
    int rv;

    /*必须要设置。*/
    rv = nghttp2_submit_settings(node_new_p->session, NGHTTP2_FLAG_NONE, iv, ARRLEN(iv));
    assert(rv == 0);

    *node = node_new;
}

static void _accept_event(abcdk_stcp_node_t *node, int *result)
{
    /*接受新的连接。*/
    *result = 0;
}

static void _connect_event(abcdk_stcp_node_t *node)
{
    h2_node_t *http_p = NULL;
    SSL *ssl_p = NULL;
    const uint8_t *ver_p;
    int ver_l;
    int chk;

    http_p = (h2_node_t *)abcdk_stcp_get_userdata(node);

    /*设置默认协议。*/
    http_p->protocol = 1;

    ssl_p = abcdk_stcp_openssl_ctx(node);
    if (!ssl_p)
        goto final;

#ifdef HEADER_SSL_H
    /*检查SSL验证结果。*/
    chk = SSL_get_verify_result(ssl_p);
    if (chk != X509_V_OK)
    {
        /*修改超时，使用超时检测器关闭。*/
        abcdk_stcp_set_timeout(node, -1);
        return;
    }

#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
    /*获取应用层协议。*/
    SSL_get0_alpn_selected(ssl_p, &ver_p, &ver_l);
    if (ver_p == NULL || ver_l <= 0)
        goto final;

    /*只区别h2版本。*/
    http_p->protocol = ((abcdk_strncmp("h2", ver_p, ABCDK_MIN(ver_l, 2), 0) == 0) ? 2 : 1);
#endif // TLSEXT_TYPE_application_layer_protocol_negotiation
#endif // HEADER_SSL_H

    

final:

    /*已连接到远端，注册读写事件。*/
    abcdk_stcp_recv_watch(node);
    abcdk_stcp_send_watch(node);
}

static void _output_event(abcdk_stcp_node_t *node)
{
    h2_node_t *http_p = NULL;

    http_p = (h2_node_t *)abcdk_stcp_get_userdata(node);

    /*把缓存数据串行化，并通过回调发送出去。*/
    nghttp2_session_send(http_p->session);
}

static void _close_event(abcdk_stcp_node_t *node)
{
}

static void _event_cb(abcdk_stcp_node_t *node, uint32_t event, int *result)
{
    switch (event)
    {
    case ABCDK_STCP_EVENT_ACCEPT:
        _accept_event(node, result);
        break;
    case ABCDK_STCP_EVENT_CONNECT:
        _connect_event(node);
        break;
    case ABCDK_STCP_EVENT_INPUT:
        break;
    case ABCDK_STCP_EVENT_OUTPUT:
        _output_event(node);
        break;
    case ABCDK_STCP_EVENT_CLOSE:
    case ABCDK_STCP_EVENT_INTERRUPT:
    default:
        _close_event(node);
        break;
    }
}

static void _request_cb(abcdk_stcp_node_t *node, const void *data, size_t size, size_t *remain)
{
    h2_node_t *http_p = NULL;

    http_p = (h2_node_t *)abcdk_stcp_get_userdata(node);

    /*默认没有剩余数据。*/
    *remain = 0;

    if (http_p->protocol == 1)
    {
        http_p->protocol = 2;
        abcdk_stcp_post_format(node,10000,
                                   "HTTP/1.1 101 Switching Protocols\r\n"
                                   "Connection: Upgrade\r\n"
                                   "Upgrade: h2c\r\n\r\n");
    }
    else
    {
        nghttp2_session_mem_recv(http_p->session,data,size);
    }

}


#ifdef HEADER_SSL_H
#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
int _test_http2_alpn_select_cb(SSL *ssl, const unsigned char **out, unsigned char *outlen,
                               const unsigned char *in, unsigned int inlen, void *arg)
{
    unsigned int srvlen;

    /*协议选择时，仅做指针的复制，因此这里要么用静态的变量，要么创建一个全局有效的。*/
    //static unsigned char srv[] = {"\x08http/1.1\x08http/1.0\x08http/0.9"};
    static unsigned char srv[] = {"\x02h2\x08http/1.1\x08http/1.0\x08http/0.9"};

    /*精确的长度。*/
    srvlen = sizeof(srv) - 1;

    /*服务端在客户端支持的协议列表中选择一个支持协议，从左到右按顺序匹配。*/
    if (SSL_select_next_proto((unsigned char **)out, outlen, in, inlen, srv, srvlen) != OPENSSL_NPN_NEGOTIATED)
    {
        return SSL_TLSEXT_ERR_ALERT_FATAL;
    }

    return SSL_TLSEXT_ERR_OK;
}

#endif // TLSEXT_TYPE_application_layer_protocol_negotiation
#endif // HEADER_SSL_H

int abcdk_test_http2(abcdk_option_t *args)
{
    abcdk_stcp_t *ctx;
    abcdk_stcp_node_t *listen_node;
    abcdk_sockaddr_t listen_addr;

    const char *cert_file = abcdk_option_get(args, "--cert-file", 0, NULL);
    const char *key_file = abcdk_option_get(args, "--key-file", 0, NULL);
    SSL_CTX *ssl_ctx = NULL;
#ifdef HEADER_SSL_H
    ssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(1,NULL,NULL,cert_file,key_file,NULL);
#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
    SSL_CTX_set_alpn_select_cb(ssl_ctx, _test_http2_alpn_select_cb, NULL);
#endif // TLSEXT_TYPE_application_layer_protocol_negotiation
#endif // HEADER_SSL_H

    ctx = abcdk_stcp_start(10, -1);
    listen_node = _node_new(ctx);

    h2_node_t *http_p = NULL;

    http_p = (h2_node_t *)abcdk_stcp_get_userdata(listen_node);

    if(http_p->ssl_ctx = ssl_ctx)
        abcdk_stcp_upgrade2openssl(listen_node,ssl_ctx,1);

    abcdk_sockaddr_from_string(&listen_addr, "0.0.0.0:3333", 0);

    abcdk_stcp_callback_t cb = {_prepare_cb, _event_cb, _request_cb};
    abcdk_stcp_listen(listen_node, &listen_addr, &cb);

    /*等待终止信号。*/
    abcdk_proc_wait_exit_signal(-1);

    return 0;
}

#else

int abcdk_test_http2(abcdk_option_t *args)
{
    return 0;
}
#endif // NGHTTP2_H
