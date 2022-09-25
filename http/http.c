/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "http/http.h"

/** HTTP连接。*/
typedef struct _abcdk_http
{
    /** 魔法数。*/
    uint32_t magic;
#define ABCDK_HTTP_MAGIC 20220920

    /**
     * 标志。
     *
     * 1：客户端。
     * 2：服务端。
     * 3：服务端(监听)。
     */
    int flag;

    /**
     * 状态。
     *
     * 0：断开或关闭。
     * 1：连接中(或监听中)。
     * 2：已连接。
     */
    volatile int status;

    /** 通知回调函数。*/
    abcdk_http_callback_t callback;

    /** 上行最大长度。*/
    size_t up_max_size;

    /** 接收缓冲区。*/
    abcdk_comm_message_t *in_buffer;

    /** 发送缓存。*/
    abcdk_comm_message_t *out_buffer;

    /** 发送队列。*/
    abcdk_comm_queue_t *out_queue;

    /*请求数据。*/
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
    abcdk_comm_message_unref(&http_p->out_buffer);
    abcdk_comm_queue_free(&http_p->out_queue);
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
    http->flag = 0;
    http->status = 1;
    http->in_buffer = NULL;
    http->out_buffer = NULL;
    http->out_queue = abcdk_comm_queue_alloc();
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

abcdk_comm_node_t *abcdk_http_alloc(abcdk_comm_t *ctx, size_t up_max_size)
{
    abcdk_comm_node_t *node = NULL;
    abcdk_http_t *http = NULL;
    abcdk_object_t *append_p = NULL;

    assert(ctx != NULL && up_max_size >= 4096);

    node = abcdk_comm_alloc(ctx);
    if (!node)
        return NULL;

    http = _abcdk_http_alloc();
    if (!http)
        goto final_error;

    /*绑定上行最大长度。*/
    http->up_max_size = up_max_size;

    append_p = abcdk_comm_append(node);
    append_p->pptrs[0] = (uint8_t *)http;
    abcdk_object_atfree(append_p, _abcdk_http_destroy_cb, NULL);
    abcdk_object_unref(&append_p);

    return node;

final_error:

    abcdk_comm_unref(&node);

    return NULL;
}

int abcdk_http_response(abcdk_comm_node_t *node, const void *data, size_t len)
{
    abcdk_http_t *http_p = NULL;
    abcdk_comm_message_t *msg = NULL;
    void *msg_ptr;
    size_t msg_len;
    int chk;

    assert(node != NULL && data != NULL && len > 0);

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_HTTP_MAGIC, "未通过http接口建立连接，不能调此接口。");

    if (!abcdk_atomic_load(&http_p->status))
        return -2;

    msg = abcdk_comm_message_alloc(len);
    if (!msg)
        goto final_error;

    msg_ptr = abcdk_comm_message_data(msg);
    msg_len = abcdk_comm_message_size(msg);

    /*复制数据。*/
    memcpy(msg_ptr, data, len);

    chk = abcdk_comm_queue_push(http_p->out_queue, msg);
    if (chk != 0)
        goto final_error;

    if (abcdk_atomic_load(&http_p->status) == 2)
        abcdk_comm_send_watch(node);

    return 0;

final_error:

    abcdk_comm_message_unref(&msg);

    return -1;
}

int abcdk_http_response2(abcdk_comm_node_t *node, abcdk_object_t *data)
{
    abcdk_http_t *http_p = NULL;
    abcdk_comm_message_t *msg = NULL;
    int chk;

    assert(node != NULL && data != NULL);

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_HTTP_MAGIC, "未通过http接口建立连接，不能调此接口。");

    if (!abcdk_atomic_load(&http_p->status))
        return -2;

    msg = abcdk_comm_message_alloc2(data);
    if (!msg)
        goto final_error;

    chk = abcdk_comm_queue_push(http_p->out_queue, msg);
    if (chk != 0)
        goto final_error;

    if (abcdk_atomic_load(&http_p->status) == 2)
        abcdk_comm_send_watch(node);

    return 0;

final_error:

    abcdk_comm_message_unref(&msg);

    return -1;
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

    /*标记为服务端。*/
    http_p->flag = 2;
    /*复制最大上行长度。*/
    http_p->up_max_size = listen_http_p->up_max_size;
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
    int chk;

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);

#ifdef HEADER_SSL_H
    /*如果SSL开启，检查SSL验证结果。*/      
    ssl_p = abcdk_comm_ssl(node);
    if(ssl_p)
    {
        chk = SSL_get_verify_result(ssl_p);
        if(chk != X509_V_OK)
        {
            /*修改超时，使用超时检测器关闭。*/
            abcdk_comm_set_timeout(node,1);
            return;
        }
    }
#endif

    /*标记已经连接。*/
    abcdk_atomic_store(&http_p->status, 2);
    /*已连接到远端，注册读写事件。*/
    abcdk_comm_recv_watch(node);
}

void _abcdk_http_event_close(abcdk_comm_node_t *node)
{
    abcdk_http_t *http_p = NULL;

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);

    /*通知应用层，连接已经关闭。*/
    if(http_p->callback.close_cb)
        http_p->callback.close_cb(node);
}

void _abcdk_http_event_input(abcdk_comm_node_t *node)
{
    abcdk_http_t *http_p = NULL;
    abcdk_http_request_t *req_p = NULL;
    void *msg_ptr;
    size_t msg_len;
    size_t msg_off;
    size_t remain;

    int chk;

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);

    /*准备接收数的缓存。*/
    if (!http_p->in_buffer)
    {
        http_p->in_buffer = abcdk_comm_message_alloc(512 * 1024);
        if (!http_p->in_buffer)
        {
            abcdk_comm_set_timeout(node, 1);
            return;
        }
    }

    chk = abcdk_comm_message_recv(node, http_p->in_buffer);
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

    /*处理接收到的数据。*/
    msg_ptr = abcdk_comm_message_data(http_p->in_buffer);
    msg_len = abcdk_comm_message_size(http_p->in_buffer);
    msg_off = abcdk_comm_message_offset(http_p->in_buffer);

    if (!http_p->request)
    {
        http_p->request = abcdk_http_request_alloc(http_p->up_max_size, NULL);
        if (!http_p->in_buffer)
        {
            abcdk_comm_set_timeout(node, 1);
            return;
        }
    }

    chk = abcdk_http_request_append(http_p->request, msg_ptr, msg_off, &remain);
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

    if (remain > 0)
    {
        /*当输入缓存中有未处理数据时，删除已处理数据，剩余数据移动到首地址。*/
        abcdk_comm_message_reset(http_p->in_buffer, msg_off - remain);
        abcdk_comm_message_drain(http_p->in_buffer);
    }
    else
    {
        abcdk_comm_message_reset(http_p->in_buffer, 0);
    }
            
    /*托管请求数据。*/
    req_p = http_p->request;
    /*请求数据已经被托管，这里不能再继续使用了。*/
    http_p->request = NULL;

    /*复用链路前要增加引用计数，以防止多线程操作同一个链路在释放回收内存后，造成应用层内存非法访问的异常。*/
    abcdk_comm_refer(node);
    /*复用链路。*/
    abcdk_comm_recv_watch(node);

    /*通知应用层，数据到达。*/
    http_p->callback.request_cb(node, req_p);

    /*删除请求数据。*/
    abcdk_http_request_unref(&req_p);
    
    /*减少引用计数。*/
    abcdk_comm_unref(&node);
}

void _abcdk_http_event_output(abcdk_comm_node_t *node)
{
    abcdk_http_t *http_p = NULL;
    int chk;

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);

NEXT_MSG:

    /*如果发送缓存是空的，则从待发送队列取出一份。*/
    if (!http_p->out_buffer)
    {
        http_p->out_buffer = abcdk_comm_queue_pop(http_p->out_queue);
        if (!http_p->out_buffer)
            return;
    }

    chk = abcdk_comm_message_send(node, http_p->out_buffer);
    if (chk < 0)
    {
        abcdk_comm_set_timeout(node, 1);
        return;
    }
    else if (chk == 0)
    {
        abcdk_comm_send_watch(node);
        return;
    }

    /*释放消息缓存，并继续发送。*/
    abcdk_comm_message_unref(&http_p->out_buffer);
    goto NEXT_MSG;
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

int abcdk_http_listen(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_http_callback_t *cb)
{
    abcdk_http_t *http_p = NULL;
    int chk;

    assert(node != NULL && addr != NULL && cb != NULL);
    ABCDK_ASSERT(cb->request_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");

    http_p = (abcdk_http_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(http_p != NULL && http_p->magic == ABCDK_HTTP_MAGIC, "未通过http接口建立连接，不能调此接口。");

    /*初始化状态，标记为监听。*/
    http_p->flag = 3;
    http_p->status = 2;
    http_p->callback = *cb;

    abcdk_comm_callback_t fcb = {_abcdk_http_prepare_cb,_abcdk_http_event_cb};
    chk = abcdk_comm_listen(node, ssl_ctx, addr, &fcb);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    return -1;
}