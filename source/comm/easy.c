/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/comm/easy.h"

/** 通讯环境。*/
typedef struct _abcdk_comm_easy
{
    /** 魔法数。*/
    uint32_t magic;
#define ABCDK_COMM_EASY_MAGIC 20221018

    /** 
     * 状态。
     * 
     * 0：断开(关闭)。
     * 1：连接中(监听中)。
     * 2：已连接。
    */
    volatile int status;
#define ABCDK_COMM_EASY_STATUS_BROKEN 0
#define ABCDK_COMM_EASY_STATUS_SYNC 1
#define ABCDK_COMM_EASY_STATUS_STABLE 2

    /** 通知回调函数。*/
    abcdk_comm_easy_callback_t callback;

    /** 接收缓冲区。*/
    abcdk_comm_message_t *in_buffer;

    /** 请求消息对象。*/
    abcdk_comm_message_t *req_msg;

    /** 发送缓存。*/
    abcdk_comm_message_t *out_buffer;

    /** 发送队列。*/
    abcdk_comm_queue_t *out_queue;

} abcdk_comm_easy_t;

void _abcdk_comm_easy_free(abcdk_comm_easy_t **easy)
{
    abcdk_comm_easy_t *easy_p = NULL;

    if (!easy || !*easy)
        return;

    easy_p = *easy;
    *easy = NULL;

    abcdk_comm_message_unref(&easy_p->in_buffer);
    abcdk_comm_message_unref(&easy_p->req_msg);
    abcdk_comm_message_unref(&easy_p->out_buffer);
    abcdk_comm_queue_free(&easy_p->out_queue);
    abcdk_heap_free(easy_p);
}

abcdk_comm_easy_t *_abcdk_comm_easy_alloc()
{
    abcdk_comm_easy_t *easy = NULL;

    easy = (abcdk_comm_easy_t *)abcdk_heap_alloc(sizeof(abcdk_comm_easy_t));
    if (!easy)
        return NULL;

    easy->magic = ABCDK_COMM_EASY_MAGIC;
    easy->status = ABCDK_COMM_EASY_STATUS_BROKEN;
    easy->in_buffer = NULL;
    easy->req_msg = NULL;
    easy->out_buffer = NULL;
    easy->out_queue = abcdk_comm_queue_alloc();

    return easy;
}

void _abcdk_comm_easy_destroy_cb(abcdk_object_t *alloc, void *opaque)
{
    abcdk_comm_easy_t *easy_p = NULL;

    if (!alloc->pptrs[0])
        return;

    easy_p = (abcdk_comm_easy_t *)alloc->pptrs[0];
    alloc->pptrs[0] = NULL;

    _abcdk_comm_easy_free(&easy_p);
}

abcdk_comm_node_t *abcdk_comm_easy_alloc(abcdk_comm_t *ctx)
{
    abcdk_comm_node_t *node = NULL;
    abcdk_comm_easy_t *easy = NULL;
    abcdk_object_t *append_p = NULL;

    assert(ctx != NULL);

    node = abcdk_comm_alloc(ctx);
    if (!node)
        return NULL;

    easy = _abcdk_comm_easy_alloc();
    if (!easy)
        goto final_error;

    append_p = abcdk_comm_append(node);
    append_p->pptrs[0] = (uint8_t *)easy;
    abcdk_object_atfree(append_p, _abcdk_comm_easy_destroy_cb, NULL);
    abcdk_object_unref(&append_p);

    return node;

final_error:

    abcdk_comm_unref(&node);

    return NULL;
}

int abcdk_comm_easy_send(abcdk_comm_node_t *node, abcdk_comm_message_t *msg)
{
    abcdk_comm_easy_t *easy_p = NULL;
    int chk;

    assert(node != NULL && msg != NULL);

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(easy_p != NULL && easy_p->magic == ABCDK_COMM_EASY_MAGIC, "未通过easy接口建立连接，不能调此接口。");

    if (!abcdk_atomic_load(&easy_p->status))
        ABCDK_ERRNO_AND_GOTO1(chk = -2, final_error);

    chk = abcdk_comm_queue_push(easy_p->out_queue, msg);
    if (chk != 0)
        ABCDK_ERRNO_AND_GOTO1(chk = -1, final_error);

    if (abcdk_atomic_load(&easy_p->status) == ABCDK_COMM_EASY_STATUS_STABLE)
        abcdk_comm_send_watch(node);

    return 0;

final_error:

    abcdk_comm_message_unref(&msg);

    return chk;
}

void _abcdk_comm_easy_prepare_cb(abcdk_comm_node_t *node, abcdk_comm_node_t *listen)
{
    abcdk_comm_easy_t *listen_comm_easy_p = NULL;
    abcdk_object_t *append_p = NULL;
    abcdk_comm_easy_t *easy_p = NULL;

    listen_comm_easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(listen);

    easy_p = _abcdk_comm_easy_alloc();
    if (!easy_p)
        return;

    append_p = abcdk_comm_append(node);
    append_p->pptrs[0] = (uint8_t *)easy_p;
    abcdk_object_atfree(append_p, _abcdk_comm_easy_destroy_cb, NULL);
    abcdk_object_unref(&append_p);

    /*复制请求回调函数指针。*/
    easy_p->callback = listen_comm_easy_p->callback;
    /*复制监听的用户环境指针。*/
    abcdk_comm_set_userdata(node, abcdk_comm_get_userdata(listen));
}

void _abcdk_comm_easy_event_accept(abcdk_comm_node_t *node, int *result)
{
    abcdk_comm_easy_t *easy_p = NULL;

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);

    if(easy_p->callback.accept_cb)
        easy_p->callback.accept_cb(node,result);
    else
        *result = 0;
}

void _abcdk_comm_easy_event_connect(abcdk_comm_node_t *node)
{
    abcdk_comm_easy_t *easy_p = NULL;
    SSL *ssl_p = NULL;
    uint32_t ver_l = 0;
    uint8_t *ver_p = NULL;
    int chk;

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);

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
    }
#endif

    /*标记已经连接。*/
    abcdk_atomic_store(&easy_p->status, ABCDK_COMM_EASY_STATUS_STABLE);
    /*已连接到远端，注册读写事件。*/
    abcdk_comm_recv_watch(node);
    abcdk_comm_send_watch(node);
}

void _abcdk_comm_easy_event_close(abcdk_comm_node_t *node)
{
    abcdk_comm_easy_t *easy_p = NULL;

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);
        
    /*标记坏了。*/
    abcdk_atomic_store(&easy_p->status, ABCDK_COMM_EASY_STATUS_BROKEN);

    /*通知应用层，连接已经关闭。*/
    if(easy_p->callback.close_cb)
        easy_p->callback.close_cb(node);
}

int _abcdk_comm_easy_event_input_unpack_cb(void *opaque, abcdk_comm_message_t *msg)
{
    abcdk_comm_node_t *node_p = NULL;
    abcdk_comm_easy_t *easy_p = NULL;
    int chk;

    node_p = (abcdk_comm_node_t *)opaque;
    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node_p);

    if(!easy_p->callback.unpack_cb)
        return 1;

    chk = easy_p->callback.unpack_cb(node_p,msg);
}

void _abcdk_comm_easy_event_input(abcdk_comm_node_t *node)
{
    abcdk_comm_easy_t *easy_p = NULL;
    abcdk_comm_message_t *req_msg_p = NULL;
    void *msg_ptr;
    size_t msg_len, msg_off, remain = 0;
    int chk;

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);

    /*准备接收数据的缓存。*/
    if (!easy_p->in_buffer)
    {
        easy_p->in_buffer = abcdk_comm_message_alloc(512 * 1024);
        if (!easy_p->in_buffer)
        {
            abcdk_comm_set_timeout(node, 1);
            return;
        }
    }

    chk = abcdk_comm_message_recv(easy_p->in_buffer,node);
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

    /*准备请求消息对象。*/
    if (!easy_p->req_msg)
    {
        easy_p->req_msg = abcdk_comm_message_alloc(1);
        if (!easy_p->req_msg)
        {
            abcdk_comm_set_timeout(node, 1);
            return;
        }

        abcdk_comm_message_protocol_t cb = {node, _abcdk_comm_easy_event_input_unpack_cb};
        abcdk_comm_message_protocol_set(easy_p->req_msg, &cb);
    }

    /*处理接收到的数据。*/
    msg_ptr = abcdk_comm_message_data(easy_p->in_buffer);
    msg_len = abcdk_comm_message_size(easy_p->in_buffer);
    msg_off = abcdk_comm_message_offset(easy_p->in_buffer);

    chk = abcdk_comm_message_recv2(easy_p->req_msg,msg_ptr,msg_off,&remain);
    
    /*从缓存中删除已处理数据。*/
    abcdk_comm_message_drain(easy_p->in_buffer, msg_off - remain);

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

    /*托管缓存。*/
    req_msg_p = easy_p->req_msg;
    /*缓存已经被托管，这里不能再继续使用了。*/
    easy_p->req_msg = NULL;

    /*通知应用层，数据到达。*/
    easy_p->callback.request_cb(node, req_msg_p);

    /*删除请求数据。*/
    abcdk_http_request_unref(&req_msg_p);

    /*如果是流水线模式，缓存中可能存在未处理的数据。*/
    if (remain > 0)
        goto NEXT_REQ;
    else
        abcdk_comm_recv_watch(node);
}

void _abcdk_comm_easy_event_output(abcdk_comm_node_t *node)
{
    abcdk_comm_easy_t *easy_p = NULL;
    int chk;

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);

NEXT_MSG:

    /*如果发送缓存是空的，则从待发送队列取出一份。*/
    if (!easy_p->out_buffer)
    {
        easy_p->out_buffer = abcdk_comm_queue_pop(easy_p->out_queue);
        if (!easy_p->out_buffer)
            return;
    }

    chk = abcdk_comm_message_send(easy_p->out_buffer,node);
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
    abcdk_comm_message_unref(&easy_p->out_buffer);
    goto NEXT_MSG;
}

void _abcdk_comm_easy_event_cb(abcdk_comm_node_t *node, uint32_t event, int *result)
{
    switch (event)
    {
    case ABCDK_COMM_EVENT_ACCEPT:
        _abcdk_comm_easy_event_accept(node,result);
        break;
    case ABCDK_COMM_EVENT_CONNECT:
        _abcdk_comm_easy_event_connect(node);
        break;
    case ABCDK_COMM_EVENT_INPUT:
        _abcdk_comm_easy_event_input(node);
        break;
    case ABCDK_COMM_EVENT_OUTPUT:
        _abcdk_comm_easy_event_output(node);
        break;
    case ABCDK_COMM_EVENT_CLOSE:
    case ABCDK_COMM_EVENT_INTERRUPT:
    default:
        _abcdk_comm_easy_event_close(node);
        break;
    }
}

int abcdk_comm_easy_listen(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_comm_easy_callback_t *cb)
{
    abcdk_comm_easy_t *easy_p = NULL;
    int chk;

    assert(node != NULL && addr != NULL && cb != NULL);
    ABCDK_ASSERT(cb->request_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(easy_p != NULL && easy_p->magic == ABCDK_COMM_EASY_MAGIC, "未通过easy接口建立连接，不能调此接口。");

    /*初始化状态。*/
    easy_p->status = ABCDK_COMM_EASY_STATUS_SYNC;
    easy_p->callback = *cb;

    abcdk_comm_callback_t fcb = {_abcdk_comm_easy_prepare_cb,_abcdk_comm_easy_event_cb};
    chk = abcdk_comm_listen(node, ssl_ctx, addr, &fcb);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    return -1;
}

int abcdk_comm_easy_connect(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_comm_easy_callback_t *cb)
{
    abcdk_comm_easy_t *easy_p = NULL;
    int chk;

    assert(node != NULL && addr != NULL && cb != NULL);
    ABCDK_ASSERT(cb->request_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(easy_p != NULL && easy_p->magic == ABCDK_COMM_EASY_MAGIC, "未通过easy接口建立连接，不能调此接口。");

    /*初始化状态。*/
    easy_p->status = ABCDK_COMM_EASY_STATUS_SYNC;
    easy_p->callback = *cb;

    abcdk_comm_callback_t fcb = {_abcdk_comm_easy_prepare_cb,_abcdk_comm_easy_event_cb};
    chk = abcdk_comm_connect(node, ssl_ctx, addr, &fcb);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    return -1;
}