/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/http/http.h"

typedef struct _abcdk_http_node
{
    /** 魔法数。*/
    uint32_t magic;
#define ABCDK_HTTP_MAGIC 123456789

    /** 通讯环境指针。*/
    abcdk_comm_t *ctx;

    /** 通知回调函数。*/
    abcdk_http_callback_t *callback;
    abcdk_http_callback_t cb_cp;

    /**
     * 协议。
     * 1: http/1.0 http/1.1 http/0.9 rtsp/1.0
     * 2: http/2
     */
    int protocol;

    /** 下层协议。*/
    int next_proto;

    /** 请求消息。*/
    abcdk_http_request_t *req;

    /** 请求最大长度。*/
    size_t req_max;

    /** 缓存目录。*/
    char req_tempdir[PATH_MAX];

} abcdk_http_node_t;

void _abcdk_http_node_destroy_cb(abcdk_object_t *obj, void *opaque)
{
    abcdk_http_node_t *http_p = NULL;

    http_p = (abcdk_http_node_t *)obj->pptrs[0];

    abcdk_http_request_unref(&http_p->req);
}

abcdk_comm_node_t *abcdk_http_alloc(abcdk_comm_t *ctx, size_t userdata, size_t max, const char *tempdir)
{
    abcdk_comm_node_t *node = NULL;
    abcdk_http_node_t *http_p = NULL;
    abcdk_object_t *extend_p = NULL;

    assert(ctx != NULL);

    node = abcdk_comm_alloc(ctx, sizeof(abcdk_http_node_t), userdata);
    if (!node)
        return NULL;

    extend_p = abcdk_comm_extend(node);
    http_p = (abcdk_http_node_t *)extend_p->pptrs[0];

    /*绑定扩展数据析构函数。*/
    abcdk_object_atfree(extend_p, _abcdk_http_node_destroy_cb, NULL);
    abcdk_object_unref(&extend_p);

    http_p->magic = ABCDK_HTTP_MAGIC;
    http_p->ctx = ctx;
    http_p->req = NULL;

    http_p->req_max = max;

    if (tempdir && *tempdir)
    {
        if (access(tempdir, W_OK) != 0)
            goto final_error;

        strncpy(http_p->req_tempdir, tempdir, PATH_MAX);
    }

    return node;

final_error:

    abcdk_comm_unref(&node);

    return NULL;
}

void _abcdk_http_prepare_cb(abcdk_comm_node_t **node, abcdk_comm_node_t *listen)
{
    abcdk_http_node_t *http_listen_p = NULL;
    abcdk_http_node_t *http_p = NULL;
    abcdk_object_t *append_p = NULL;
    abcdk_comm_node_t *node_p = NULL;

    http_listen_p = (abcdk_http_node_t *)abcdk_comm_get_extend0(listen);

    /*如果未指定准备函数，则准备基本的节点环境。*/
    if (http_listen_p->callback->prepare_cb)
        http_listen_p->callback->prepare_cb(&node_p, listen);
    else
        node_p = abcdk_http_alloc(http_listen_p->ctx, 0, http_listen_p->req_max, http_listen_p->req_tempdir);

    if (!node_p)
        return;

    http_p = (abcdk_http_node_t *)abcdk_comm_get_extend0(node_p);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_HTTP_MAGIC, "未通过HTTP接口建立连接，不能调此接口。");

    /*复制指针。*/
    http_p->callback = http_listen_p->callback;

    /*设置下层协议。*/
    http_p->next_proto = ABCDK_HTTP_REQUEST_PROTO_NATURAL;

    /*准备完毕，返回。*/
    *node = node_p;
}

void _abcdk_http_accept_cb(abcdk_comm_node_t *node, int *result)
{
    abcdk_http_node_t *http_p = NULL;

    http_p = (abcdk_http_node_t *)abcdk_comm_get_extend0(node);

    if (http_p->callback->accept_cb)
        http_p->callback->accept_cb(node, result);
    else
        *result = 0;
}

void _abcdk_http_connect_cb(abcdk_comm_node_t *node)
{
    abcdk_http_node_t *http_p = NULL;
    SSL *ssl_p = NULL;
    void *ver_p;
    int ver_l;
    int chk;

    http_p = (abcdk_http_node_t *)abcdk_comm_get_extend0(node);

    /*默认协议。*/
    http_p->protocol = 1;

#ifdef HEADER_SSL_H
    /*如果SSL开启，检查SSL验证结果。*/
    ssl_p = abcdk_comm_ssl(node);
    if (ssl_p)
    {
        chk = SSL_get_verify_result(ssl_p);
        if (chk != X509_V_OK)
        {
            /*修改超时，使用超时检测器关闭。*/
            abcdk_comm_set_timeout(node, 1);
            return;
        }

#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
        SSL_get0_alpn_selected(ssl_p, (const uint8_t **)&ver_p, &ver_l);
        if (ver_p != NULL && ver_l > 0)
            http_p->protocol = ((abcdk_strncmp("h2", ver_p, ABCDK_MIN(ver_l, 2), 0) == 0) ? 2 : 1);
#endif // TLSEXT_TYPE_application_layer_protocol_negotiation
    }
#endif

    /*已连接到远端，注册读写事件。*/
    abcdk_comm_recv_watch(node);
    abcdk_comm_send_watch(node);
}

void _abcdk_http_input_cb(abcdk_comm_node_t *node)
{
    abcdk_http_node_t *http_p = NULL;

    http_p = (abcdk_http_node_t *)abcdk_comm_get_extend0(node);
}

void _abcdk_http_output_cb(abcdk_comm_node_t *node)
{
    abcdk_http_node_t *http_p = NULL;

    http_p = (abcdk_http_node_t *)abcdk_comm_get_extend0(node);

    if (http_p->callback->output_cb)
        http_p->callback->output_cb(node);
}

void _abcdk_http_close_cb(abcdk_comm_node_t *node)
{
    abcdk_http_node_t *http_p = NULL;

    http_p = (abcdk_http_node_t *)abcdk_comm_get_extend0(node);

    /*通知连接已断开。*/
    if (http_p->callback->close_cb)
        http_p->callback->close_cb(node);
}

void _abcdk_http_event_cb(abcdk_comm_node_t *node, uint32_t event, int *result)
{
    switch (event)
    {
    case ABCDK_COMM_EVENT_ACCEPT:
        _abcdk_http_accept_cb(node, result);
        break;
    case ABCDK_COMM_EVENT_CONNECT:
        _abcdk_http_connect_cb(node);
        break;
    case ABCDK_COMM_EVENT_INPUT:
        break;
    case ABCDK_COMM_EVENT_OUTPUT:
        _abcdk_http_output_cb(node);
        break;
    case ABCDK_COMM_EVENT_CLOSE:
    case ABCDK_COMM_EVENT_INTERRUPT:
    default:
        _abcdk_http_close_cb(node);
        break;
    }
}

void _abcdk_http_request_v1(abcdk_comm_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_http_node_t *http_p;
    int chk;

    http_p = (abcdk_http_node_t *)abcdk_comm_get_extend(node);

    if (!http_p->req)
        http_p->req = abcdk_http_request_alloc(http_p->next_proto, http_p->req_max, http_p->req_tempdir);

    if (!http_p->req)
    {
        abcdk_comm_set_timeout(node, 1);
        return;
    }

    chk = abcdk_http_request_append(http_p->req, data, size, remain);
    if (chk < 0)
    {
        abcdk_comm_set_timeout(node, 1);
        return;
    }
    else if (chk == 0)
    {
        /*数据包不完整，继接收请求数据。*/
        abcdk_comm_recv_watch(node);
        return;
    }
    else if (chk > 0)
    {
        if (http_p->callback->request_cb)
            http_p->callback->request_cb(node, http_p->req, &http_p->next_proto);

        abcdk_http_request_unref(&http_p->req);

        /*隧道协议，是否继续接收由应用层决定。*/
        if (http_p->next_proto != ABCDK_HTTP_REQUEST_PROTO_TUNNEL)
            abcdk_comm_recv_watch(node);
    }
}

void _abcdk_http_request_cb(abcdk_comm_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_http_node_t *http_p;
    int chk;

    http_p = (abcdk_http_node_t *)abcdk_comm_get_extend(node);

    /*默认处理全部数据。*/
    *remain = 0;

    if (http_p->protocol == 1)
        _abcdk_http_request_v1(node, data, size, remain);
    else
        abcdk_comm_set_timeout(node, 1);
}

int abcdk_http_listen(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_http_callback_t *cb)
{
    abcdk_http_node_t *http_p = NULL;
    int chk;

    assert(node != NULL && addr != NULL && cb != NULL);
    ABCDK_ASSERT(cb->request_cb != NULL, "未绑定通知回调函数，通讯对象无法正常工作。");

    http_p = (abcdk_http_node_t *)abcdk_comm_get_extend0(node);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_HTTP_MAGIC, "未通过HTTP接口建立连接，不能调此接口。");

    /*初始化状态。*/
    http_p->cb_cp = *cb;
    http_p->callback = &http_p->cb_cp;

    abcdk_comm_callback_t fcb = {_abcdk_http_prepare_cb, _abcdk_http_event_cb, _abcdk_http_request_cb};
    chk = abcdk_comm_listen(node, ssl_ctx, addr, &fcb);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    return -1;
}


int abcdk_http_connect(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_http_callback_t *cb)
{
    abcdk_http_node_t *http_p = NULL;
    int chk;

    assert(node != NULL && addr != NULL && cb != NULL);
    ABCDK_ASSERT(cb->request_cb != NULL, "未绑定通知回调函数，通讯对象无法正常工作。");

    http_p = (abcdk_http_node_t *)abcdk_comm_get_extend0(node);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_HTTP_MAGIC, "未通过HTTP接口建立连接，不能调此接口。");

    /*初始化状态。*/
    http_p->cb_cp = *cb;
    http_p->callback = &http_p->cb_cp;
    http_p->next_proto = ABCDK_HTTP_REQUEST_PROTO_TUNNEL;

    abcdk_comm_callback_t fcb = {_abcdk_http_prepare_cb, _abcdk_http_event_cb, _abcdk_http_request_cb};
    chk = abcdk_comm_connect(node, ssl_ctx, addr, &fcb);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    return -1;
}