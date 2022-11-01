/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/http/http.h"

/** HTTP连接。*/
typedef struct _abcdk_http
{
    /** 魔法数。*/
    uint32_t magic;
#define ABCDK_HTTP_MAGIC 20220920

    /** 
     * 状态。
     * 
     * 0：断开(关闭)。
     * 1：连接中(监听中)。
     * 2：已连接。
    */
    volatile int status;
#define ABCDK_HTTP_STATUS_BROKEN 0
#define ABCDK_HTTP_STATUS_SYNC 1
#define ABCDK_HTTP_STATUS_STABLE 2

    /**
     * 0: unknown
     * 1: http/1.0 or http/1.1 or http/0.9 or rtsp/1.0
     * 2: http/2
    */
    int version;

    /** 通知回调函数。*/
    abcdk_http_callback_t callback;

    /** 上行最大长度。*/
    size_t up_max_size;

    /** 上行实体缓存目录。*/
    char up_buffer_point[PATH_MAX];

    /** 接收缓冲区。*/
    abcdk_comm_message_t *in_buffer;

    /** 请求数据。*/
    abcdk_http_request_t *request;

} abcdk_http_t;

void _abcdk_http_free(abcdk_http_t **http)
{
    abcdk_http_t *http_p = NULL;

    if (!http || !*http)
        return;

    http_p = *http;
    *http = NULL;

    abcdk_comm_message_unref(&http_p->in_buffer);
    abcdk_http_request_unref(&http_p->request);
    abcdk_heap_free(http_p);
}

abcdk_http_t *_abcdk_http_alloc()
{
    abcdk_http_t *http = NULL;

    http = (abcdk_http_t *)abcdk_heap_alloc(sizeof(abcdk_http_t));
    if (!http)
        return NULL;

    http->magic = ABCDK_HTTP_MAGIC;
    http->status = ABCDK_HTTP_STATUS_BROKEN;
    http->version = 1;
    http->up_max_size = 40960;
    memset(http->up_buffer_point,0,PATH_MAX);
    http->in_buffer = NULL;
    http->request = NULL;

    return http;
}

void _abcdk_http_destroy_cb(abcdk_object_t *alloc, void *opaque)
{
    abcdk_http_t *http_p = NULL;

    if (!alloc->pptrs[0])
        return;

    http_p = (abcdk_http_t *)alloc->pptrs[0];
    alloc->pptrs[0] = NULL;

    _abcdk_http_free(&http_p);
}

abcdk_comm_node_t *abcdk_http_alloc(abcdk_comm_t *ctx, size_t up_max_size,const char *up_buffer_point)
{
    abcdk_comm_node_t *node = NULL;
    abcdk_http_t *http = NULL;
    abcdk_object_t *append_p = NULL;

    assert(ctx != NULL && up_max_size >= 4096);
    assert(up_buffer_point == NULL || (up_buffer_point != NULL && strlen(up_buffer_point) <= PATH_MAX - 6));

    node = abcdk_comm_alloc(ctx);
    if (!node)
        return NULL;

    http = _abcdk_http_alloc();
    if (!http)
        goto final_error;

    http->up_max_size = up_max_size;
    if (up_buffer_point)
        strncpy(http->up_buffer_point, up_buffer_point, PATH_MAX);

    append_p = abcdk_comm_append(node);
    append_p->pptrs[0] = (uint8_t *)http;
    abcdk_object_atfree(append_p, _abcdk_http_destroy_cb, NULL);
    abcdk_object_unref(&append_p);

    return node;

final_error:

    abcdk_comm_unref(&node);

    return NULL;
}

void _abcdk_http_prepare_cb(abcdk_comm_node_t *node, abcdk_comm_node_t *listen)
{
    abcdk_http_t *listen_http_p = NULL;
    abcdk_object_t *append_p = NULL;
    abcdk_http_t *http_p = NULL;

    listen_http_p = (abcdk_http_t *)abcdk_comm_get_append(listen);

    http_p = _abcdk_http_alloc();
    if (!http_p)
        return;

    append_p = abcdk_comm_append(node);
    append_p->pptrs[0] = (uint8_t *)http_p;
    abcdk_object_atfree(append_p, _abcdk_http_destroy_cb, NULL);
    abcdk_object_unref(&append_p);

    /*复制最大上行长度。*/
    http_p->up_max_size = listen_http_p->up_max_size;
    /*复制上行实体缓存目录。*/
    if (listen_http_p->up_buffer_point[0])
        strncpy(http_p->up_buffer_point, listen_http_p->up_buffer_point, PATH_MAX);
    /*复制请求回调函数指针。*/
    http_p->callback = listen_http_p->callback;
    /*复制监听的用户环境指针。*/
    abcdk_comm_set_userdata(node, abcdk_comm_get_userdata(listen));
}

void _abcdk_http_event_accept(abcdk_comm_node_t *node, int *result)
{
    abcdk_http_t *http_p = NULL;

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);

    if(http_p->callback.accept_cb)
        http_p->callback.accept_cb(node,result);
    else
        *result = 0;
}

void _abcdk_http_event_connect(abcdk_comm_node_t *node)
{
    abcdk_http_t *http_p = NULL;
    SSL *ssl_p = NULL;
    uint32_t ver_l = 0;
    uint8_t *ver_p = NULL;
    int chk;

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);

    /*默认支持1.1 or 1.0 or 0.9。*/
    http_p->version = 1;

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
        SSL_get0_alpn_selected(ssl_p, (const uint8_t**)&ver_p, &ver_l);
        if (ver_p != NULL && ver_l > 0)
            http_p->version = ((abcdk_strncmp("h2", ver_p, ABCDK_MIN(ver_l, 2),0) == 0) ? 2 : 1);
#endif //TLSEXT_TYPE_application_layer_protocol_negotiation

    }
#endif //HEADER_SSL_H

    /*标记已经连接。*/
    abcdk_atomic_store(&http_p->status, ABCDK_HTTP_STATUS_STABLE);
    /*已连接到远端，注册读写事件。*/
    abcdk_comm_recv_watch(node);
    //abcdk_comm_send_watch(node);
}

void _abcdk_http_event_close(abcdk_comm_node_t *node)
{
    abcdk_http_t *http_p = NULL;

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);
        
    /*标记坏了。*/
    abcdk_atomic_store(&http_p->status, ABCDK_HTTP_STATUS_BROKEN);

    /*通知应用层，连接已经关闭。*/
    if(http_p->callback.close_cb)
        http_p->callback.close_cb(node);
}

void _abcdk_http_event_input_v1(abcdk_comm_node_t *node)
{
    abcdk_http_t *http_p = NULL;
    void *msg_ptr;
    size_t msg_len, msg_off, remain = 0;
    int chk;

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);

    /*准备接收数据的缓存。*/
    if (!http_p->in_buffer)
    {
        http_p->in_buffer = abcdk_comm_message_alloc(512 * 1024);
        if (!http_p->in_buffer)
        {
            abcdk_comm_set_timeout(node, 1);
            return;
        }
    }

    chk = abcdk_comm_message_recv(http_p->in_buffer,node);
    if (chk < 0)
    {
        abcdk_comm_set_timeout(node, 1);
        return;
    }
    else if (chk == 0)
    {
        abcdk_comm_recv_watch(node);
        return;
    }

NEXT_REQ:

    /*处理接收到的数据。*/
    msg_ptr = abcdk_comm_message_data(http_p->in_buffer);
    msg_len = abcdk_comm_message_size(http_p->in_buffer);
    msg_off = abcdk_comm_message_offset(http_p->in_buffer);
    
    /*准备请求环境。*/
    if (!http_p->request)
    {
        http_p->request = abcdk_http_request_alloc(http_p->up_max_size, http_p->up_buffer_point);
        if (!http_p->request)
        {
            abcdk_comm_set_timeout(node, 1);
            return;
        }
    }

    /*填加到请求环境对象。*/
    chk = abcdk_http_request_append(http_p->request, msg_ptr, msg_off, &remain);

    /*从缓存中删除已处理数据。*/
    abcdk_comm_message_drain(http_p->in_buffer, msg_off - remain);

    if (chk < 0)
    {
        abcdk_comm_set_timeout(node, 1);
        return;
    }
    else if (chk == 0)
    {
        abcdk_comm_recv_watch(node);
        return;
    }

    /*通知应用层，数据到达。*/
    http_p->callback.request_cb(node, http_p->request);

    /*删除请求数据。*/
    abcdk_http_request_unref(&http_p->request);

    /*如果是流水线模式，缓存中可能存在未处理的数据。*/
    if (remain > 0)
        goto NEXT_REQ;
    else
        abcdk_comm_recv_watch(node);
}

void _abcdk_http_event_input(abcdk_comm_node_t *node)
{
    abcdk_http_t *http_p = NULL;
    int chk;

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);

    if(http_p->version == 1)
        _abcdk_http_event_input_v1(node);
    else 
        abcdk_comm_set_timeout(node, 1);
}

void _abcdk_http_event_output(abcdk_comm_node_t *node)
{
    abcdk_http_t *http_p = NULL;
    int chk;

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);

    /*如果应用层需要，通知发送队列空闲。*/
    if(http_p->callback.fetch_cb)
        http_p->callback.fetch_cb(node);
}

void _abcdk_http_event_cb(abcdk_comm_node_t *node, uint32_t event, int *result)
{
    switch (event)
    {
    case ABCDK_COMM_EVENT_ACCEPT:
        _abcdk_http_event_accept(node,result);
        break;
    case ABCDK_COMM_EVENT_CONNECT:
        _abcdk_http_event_connect(node);
        break;
    case ABCDK_COMM_EVENT_INPUT:
        _abcdk_http_event_input(node);
        break;
    case ABCDK_COMM_EVENT_OUTPUT:
        _abcdk_http_event_output(node);
        break;
    case ABCDK_COMM_EVENT_CLOSE:
    case ABCDK_COMM_EVENT_INTERRUPT:
    default:
        _abcdk_http_event_close(node);
        break;
    }
}

#ifdef HEADER_SSL_H
#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
int _abcdk_http_alpn_select_cb(SSL *ssl, const unsigned char **out, unsigned char *outlen,
                               const unsigned char *in, unsigned int inlen, void *arg)
{
    unsigned int srvlen;

    /*协议选择时，仅做指针的复制，因此这里要么用静态的变量，要么创建一个全局有效的。*/
    static unsigned char srv[] = {"\x08http/1.1\x08http/1.0\x08http/0.9"};

    /*精确的长度。*/
    srvlen = sizeof(srv) - 1;

    /*服务端在客户端支持的协议列表中选择一个支持协议，从左到右按顺序匹配。*/
    if (SSL_select_next_proto((unsigned char **)out, outlen, srv, srvlen, in, inlen) != OPENSSL_NPN_NEGOTIATED)
    {
        return SSL_TLSEXT_ERR_ALERT_FATAL;
    }

    return SSL_TLSEXT_ERR_OK;
}

#endif //TLSEXT_TYPE_application_layer_protocol_negotiation
#endif //HEADER_SSL_H

int abcdk_http_listen(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_http_callback_t *cb)
{
    abcdk_http_t *http_p = NULL;
    int chk;

    assert(node != NULL && addr != NULL && cb != NULL);
    ABCDK_ASSERT(cb->request_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_HTTP_MAGIC, "未通过http接口建立连接，不能调此接口。");

    /*初始化状态。*/
    http_p->status = ABCDK_HTTP_STATUS_SYNC;
    http_p->callback = *cb;

#ifdef HEADER_SSL_H
#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
    if(ssl_ctx)
        SSL_CTX_set_alpn_select_cb(ssl_ctx, _abcdk_http_alpn_select_cb, NULL);
#endif //TLSEXT_TYPE_application_layer_protocol_negotiation
#endif //HEADER_SSL_H

    abcdk_comm_callback_t fcb = {_abcdk_http_prepare_cb,_abcdk_http_event_cb};
    chk = abcdk_comm_listen(node, ssl_ctx, addr, &fcb);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    return -1;
}