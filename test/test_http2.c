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

#ifdef HAVE_NGHTTP2
#include <nghttp2/nghttp2.h>
#endif // HAVE_NGHTTP2

#ifdef NGHTTP2_H

typedef struct _h2_node
{
    abcdk_asynctcp_t *ctx;

    int protocol;

    nghttp2_session_callbacks *callbacks;
    nghttp2_session *session;
} h2_node_t;

// nghttp2回调函数，用于处理HTTP/2会话事件
static int on_data_chunk_recv_callback(nghttp2_session *session, uint8_t flags, int32_t stream_id, const uint8_t *data, size_t len, void *user_data)
{
    // 在这里处理接收到的数据
    // 这里只是简单地打印接收到的数据
    printf("Received data on stream %d: %.*s\n", stream_id, (int)len, data);

    // 返回成功
    return 0;
}

// nghttp2回调函数，用于处理HTTP/2帧头部
static int on_header_callback(nghttp2_session *session, const nghttp2_frame *frame, const uint8_t *name, size_t namelen, const uint8_t *value, size_t valuelen, uint8_t flags, void *user_data)
{
    // 在这里处理帧头部
    // 这里只是简单地打印头部信息
    printf("Received header on stream %d: %.*s: %.*s\n", frame->hd.stream_id, (int)namelen, name, (int)valuelen, value);

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

static void _node_destroy_cb(abcdk_object_t *obj, void *opaque)
{
    h2_node_t *http_p;

    http_p = (h2_node_t *)obj->pptrs[0];
    if (!http_p)
        return;

    if (http_p->session)
        nghttp2_session_del(http_p->session);
    if (http_p->callbacks)
        nghttp2_session_callbacks_del(http_p->callbacks);
}

static abcdk_asynctcp_node_t *_node_new(abcdk_asynctcp_t *ctx)
{
    abcdk_object_t *userdata_p = NULL;
    h2_node_t *http_p = NULL;
    abcdk_asynctcp_node_t *node_new;

    node_new = abcdk_asynctcp_alloc(ctx, sizeof(h2_node_t));

    userdata_p = abcdk_asynctcp_userdata(node_new);
    http_p = (h2_node_t *)userdata_p->pptrs[0];

    abcdk_object_atfree(userdata_p, _node_destroy_cb, NULL);
    abcdk_object_unref(&userdata_p);

    http_p->ctx = ctx;

    return node_new;
}

static void _prepare_cb(abcdk_asynctcp_node_t **node, abcdk_asynctcp_node_t *listen)
{
    h2_node_t *listen_p = NULL, *node_new_p = NULL;
    abcdk_asynctcp_node_t *node_new;

    listen_p = (h2_node_t *)abcdk_asynctcp_get_userdata(listen);

    node_new = _node_new(listen_p->ctx);
    node_new_p = (h2_node_t *)abcdk_asynctcp_get_userdata(node_new);

    // 初始化nghttp2回调结构
    nghttp2_session_callbacks_new(&node_new_p->callbacks);
    nghttp2_session_callbacks_set_on_data_chunk_recv_callback(node_new_p->callbacks, on_data_chunk_recv_callback);
    nghttp2_session_callbacks_set_on_header_callback(node_new_p->callbacks, on_header_callback);
    nghttp2_session_callbacks_set_on_stream_close_callback(node_new_p->callbacks, on_stream_close_callback);

    // 创建nghttp2服务器会话
    nghttp2_session_server_new(&node_new_p->session, node_new_p->callbacks, NULL);

    // nghttp2_session_set_stream_user_data(node_new_p->session, 1, bev);

    *node = node_new;
}

static void _accept_event(abcdk_asynctcp_node_t *node, int *result)
{
    /*接受新的连接。*/
    *result = 0;
}

static void _connect_event(abcdk_asynctcp_node_t *node)
{
    h2_node_t *http_p = NULL;

    http_p = (h2_node_t *)abcdk_asynctcp_get_userdata(node);

    http_p->protocol = 1;

    /*已连接到远端，注册读写事件。*/
    abcdk_asynctcp_recv_watch(node);
    abcdk_asynctcp_send_watch(node);
}

static void _output_event(abcdk_asynctcp_node_t *node)
{
}

static void _close_event(abcdk_asynctcp_node_t *node)
{
}

static void _event_cb(abcdk_asynctcp_node_t *node, uint32_t event, int *result)
{
    switch (event)
    {
    case ABCDK_ASYNCTCP_EVENT_ACCEPT:
        _accept_event(node, result);
        break;
    case ABCDK_ASYNCTCP_EVENT_CONNECT:
        _connect_event(node);
        break;
    case ABCDK_ASYNCTCP_EVENT_INPUT:
        break;
    case ABCDK_ASYNCTCP_EVENT_OUTPUT:
        _output_event(node);
        break;
    case ABCDK_ASYNCTCP_EVENT_CLOSE:
    case ABCDK_ASYNCTCP_EVENT_INTERRUPT:
    default:
        _close_event(node);
        break;
    }
}

static void _request_cb(abcdk_asynctcp_node_t *node, const void *data, size_t size, size_t *remain)
{
    h2_node_t *http_p = NULL;

    http_p = (h2_node_t *)abcdk_asynctcp_get_userdata(node);

    /*默认没有剩余数据。*/
    *remain = 0;

    if (http_p->protocol == 1)
    {
        http_p->protocol = 2;
        abcdk_asynctcp_post_format(node,10000,
                                   "HTTP/1.1 101 Switching Protocols\r\n"
                                   "Connection: Upgrade\r\n"
                                   "Upgrade: h2c\r\n\r\n");
    }
    else
    {
        nghttp2_session_mem_recv(http_p->session,data,size);
    }

}

int abcdk_test_http2(abcdk_option_t *args)
{
    abcdk_asynctcp_t *ctx;
    abcdk_asynctcp_node_t *listen_node;
    abcdk_sockaddr_t listen_addr;

    ctx = abcdk_asynctcp_start(10, -1);
    listen_node = _node_new(ctx);

    abcdk_sockaddr_from_string(&listen_addr, "0.0.0.0:3333", 0);

    abcdk_asynctcp_callback_t cb = {_prepare_cb, _event_cb, _request_cb};
    abcdk_asynctcp_listen(listen_node, NULL, &listen_addr, &cb);

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
