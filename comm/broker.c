/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-comm/broker.h"

/**通信节点。*/
typedef struct _abcdk_broker_node
{
    /** 引用计数器。*/
    volatile int refcount;

    /** 链路状态。*/
    volatile int stable;

    /** 通信链路。*/
    abcdk_comm_node_t *comm;

    /** 消息回调。*/
    abcdk_broker_message_cb message_cb;

    /*应用层环境指针。*/
    void *opaque;

    /** 接收缓存区。*/
    abcdk_tree_t *in_buffer;

    /** 发送缓存区。*/
    abcdk_tree_t *out_buffer;

    /** 发送队列锁。*/
    abcdk_mutex_t out_locker;

    /** 发送队列。*/
    abcdk_tree_t *out_queue;

} abcdk_broker_node_t;

void abcdk_broker_node_unref(abcdk_broker_node_t **node)
{
    abcdk_broker_node_t *node_p;

    if (!node || !*node)
        return;

    node_p = *node;

    if (abcdk_atomic_fetch_and_add(&node_p->refcount, -1) != 1)
        goto final;

    assert(node_p->refcount == 0);

    abcdk_tree_free(&node_p->in_buffer);
    abcdk_tree_free(&node_p->out_buffer);
    abcdk_tree_free(&node_p->out_queue);
    abcdk_mutex_destroy(&node_p->out_locker);
    abcdk_comm_node_unref(&node_p->comm);

    abcdk_heap_free(node_p);

final:

    /*set NULL(0).*/
    *node = NULL;
}

abcdk_broker_node_t *abcdk_broker_node_refer(abcdk_broker_node_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_broker_node_t *_abcdk_broker_node_alloc()
{
    abcdk_broker_node_t *node;

    node = abcdk_heap_alloc(sizeof(abcdk_broker_node_t));
    if (!node)
        return NULL;

    node->refcount = 1;
    node->stable = 1;
    node->comm = NULL;
    node->message_cb = NULL;
    node->opaque = NULL;

    node->in_buffer = NULL;
    node->out_buffer = NULL;

    abcdk_mutex_init2(&node->out_locker, 0);
    node->out_queue = abcdk_tree_alloc3(1);
    if (!node->out_queue)
        goto final_error;

    return node;

final_error:

    abcdk_broker_node_unref(&node);

    return NULL;
}

void _abcdk_broker_msg_destroy_cb(abcdk_allocator_t *alloc, void *opaque)
{
    abcdk_comm_msg_t *msg_p = NULL;

    msg_p = (abcdk_comm_msg_t *)alloc->pptrs[0];

    abcdk_comm_msg_unref(&msg_p);
}

abcdk_tree_t *_abcdk_broker_msg_alloc(size_t size)
{
    abcdk_tree_t *msg = NULL;

    msg = abcdk_tree_alloc3(0);
    if (!msg)
        return NULL;

    abcdk_allocator_atfree(msg->alloc, _abcdk_broker_msg_destroy_cb, NULL);

    if (size > 0)
    {
        msg->alloc->pptrs[0] = (uint8_t *)abcdk_comm_msg_alloc(size);
        if (!msg->alloc->pptrs[0])
            goto final_error;
    }

    return msg;

final_error:

    abcdk_tree_free(&msg);

    return NULL;
}

abcdk_tree_t *_abcdk_broker_msg_alloc2(abcdk_comm_msg_t *msg)
{
    abcdk_tree_t *msg_node;

    msg_node = _abcdk_broker_msg_alloc(0);
    if (!msg_node)
        return NULL;

    msg_node->alloc->pptrs[0] = (uint8_t *)msg;

    return msg_node;
}

int _abcdk_broker_out_push(abcdk_broker_node_t *node, abcdk_comm_msg_t *msg)
{
    abcdk_tree_t *msg_node;

    if (msg)
    {
        msg_node = _abcdk_broker_msg_alloc2(msg);
        if (!msg_node)
            return -1;

        abcdk_mutex_lock(&node->out_locker, 1);
        abcdk_tree_insert2(node->out_queue, msg_node, 0);
        abcdk_mutex_unlock(&node->out_locker);
    }

    if (abcdk_atomic_load(&node->stable) == 2)
        abcdk_comm_write_watch(node->comm);

    return 0;
}

abcdk_tree_t *_abcdk_broker_out_pop(abcdk_broker_node_t *node)
{
    abcdk_tree_t *msg_node = NULL;

    abcdk_mutex_lock(&node->out_locker, 1);
    msg_node = abcdk_tree_child(node->out_queue, 1);
    if (msg_node)
        abcdk_tree_unlink(msg_node);
    abcdk_mutex_unlock(&node->out_locker);

    return msg_node;
}

void _abcdk_broker_accept_event(abcdk_comm_node_t *comm)
{
    abcdk_broker_node_t *node_listen;
    abcdk_broker_node_t *node_accpet;

    node_accpet = _abcdk_broker_node_alloc();
    if (!node_accpet)
    {
        abcdk_comm_set_timeout(comm, 1);
        return;
    }

    node_listen = (abcdk_broker_node_t *)abcdk_comm_get_userdata(comm);
    if (!node_listen)
    {
        abcdk_comm_set_timeout(comm, 1);
        return;
    }

    node_accpet->message_cb = node_listen->message_cb;
    node_accpet->opaque = node_listen->opaque;
    abcdk_atomic_store(&node_accpet->stable,2);
    node_accpet->comm = abcdk_comm_node_refer(comm);

    /*通过listen节点建立的连接，初始环境是listen节点的环境指针，现在替换成私有环境指针。*/
    abcdk_comm_set_userdata(comm, node_accpet);

    /*默认30秒超时。*/
    abcdk_comm_set_timeout(comm, 30000);

    /*监听接收事件。*/
    abcdk_comm_read_watch(comm, 0);
}

void _abcdk_broker_connect_event(abcdk_comm_node_t *comm)
{
    abcdk_broker_node_t *node;
    int chk;

    node = (abcdk_broker_node_t *)abcdk_comm_get_userdata(comm);
    if (!node)
    {
        abcdk_comm_set_timeout(comm, 1);
        return;
    }

    abcdk_atomic_store(&node->stable,2);
    node->comm = abcdk_comm_node_refer(comm);

    /*默认30秒超时。*/
    abcdk_comm_set_timeout(comm, 30000);

    /*监听接收事件。*/
    abcdk_comm_read_watch(comm, 0);

    /*可能有需要发送的已经在排队中。*/
    _abcdk_broker_out_push(node, NULL);
}

void _abcdk_broker_input_event(abcdk_comm_node_t *comm)
{
    abcdk_broker_node_t *node;
    abcdk_tree_t *msg_in;
    abcdk_comm_msg_t *msg_in_p;
    int chk;

    node = (abcdk_broker_node_t *)abcdk_comm_get_userdata(comm);
    if (!node)
    {
        abcdk_comm_set_timeout(comm, 1);
        return;
    }

    if (!node->in_buffer)
    {
        node->in_buffer = _abcdk_broker_msg_alloc(64);
        if (!node->in_buffer)
        {
            abcdk_comm_set_timeout(comm, 1);
            return;
        }
    }

    msg_in = node->in_buffer;
    msg_in_p = (abcdk_comm_msg_t *)msg_in->alloc->pptrs[0];

    chk = abcdk_comm_msg_recv(comm, msg_in_p);
    if (chk < 0)
    {
        abcdk_comm_set_timeout(comm, 1);
    }
    else if (chk == 0)
    {
        /*数据包不完整，继续接收。*/
        abcdk_comm_read_watch(comm, 1);
    }
    else if (chk > 0)
    {
        /*接收缓存区复位，链路复用。*/
        node->in_buffer = NULL;
        abcdk_comm_read_watch(comm, 1);

        /*通知应用层消息达到。*/
        node->message_cb(node, msg_in_p, node->opaque);

        /*删除已经处理的消息。*/
        abcdk_tree_free(&msg_in);
    }
}

void _abcdk_broker_output_event(abcdk_comm_node_t *comm)
{
    abcdk_broker_node_t *node;
    abcdk_tree_t *msg_out;
    abcdk_comm_msg_t *msg_out_p;
    int chk;

    node = (abcdk_broker_node_t *)abcdk_comm_get_userdata(comm);
    if (!node)
    {
        abcdk_comm_set_timeout(comm, 1);
        return;
    }

NEXT_MSG:

    msg_out = NULL;
    msg_out_p = NULL;

    if (!node->out_buffer)
    {
        node->out_buffer = _abcdk_broker_out_pop(node);
        if (!node->out_buffer)
            return;
    }

    msg_out = node->out_buffer;
    msg_out_p = (abcdk_comm_msg_t *)msg_out->alloc->pptrs[0];

    chk = abcdk_comm_msg_send(comm, msg_out_p);
    if (chk < 0)
    {
        abcdk_comm_set_timeout(comm, 1);
        return;
    }
    else if (chk > 0)
    {
        /*释放消息缓存，并继续发送。*/
        abcdk_tree_free(&node->out_buffer);
        goto NEXT_MSG;
    }

    abcdk_comm_write_watch(comm);
}

void _abcdk_broker_close_event(abcdk_comm_node_t *comm)
{
    abcdk_broker_node_t *node;

    node = (abcdk_broker_node_t *)abcdk_comm_get_userdata(comm);
    if (!node)
        return;

    abcdk_atomic_store(&node->stable,0);

    /*通知应用层连接已关闭。*/
    node->message_cb(node, NULL, node->opaque);

    abcdk_broker_node_unref(&node);
}

void _abcdk_broker_listen_close_event(abcdk_comm_node_t *comm)
{
    abcdk_broker_node_t *node;

    node = (abcdk_broker_node_t *)abcdk_comm_get_userdata(comm);
    if (!node)
        return;

    abcdk_broker_node_unref(&node);
}

void _abcdk_broker_event_cb(abcdk_comm_node_t *node, uint32_t event)
{
    switch (event)
    {
    case ABCDK_COMM_EVENT_ACCEPT:
    {
        _abcdk_broker_accept_event(node);
    }
    break;
    case ABCDK_COMM_EVENT_CONNECT:
    {
        _abcdk_broker_connect_event(node);
    }
    break;
    case ABCDK_COMM_EVENT_INPUT:
    {
        _abcdk_broker_input_event(node);
    }
    break;
    case ABCDK_COMM_EVENT_OUTPUT:
    {
        _abcdk_broker_output_event(node);
    }
    break;
    case ABCDK_COMM_EVENT_CLOSE:
    {
        _abcdk_broker_close_event(node);
    }
    break;
    case ABCDK_COMM_EVENT_LISTEN_CLOSE:
    {
        _abcdk_broker_listen_close_event(node);
    }
    break;
    }
}

int abcdk_broker_set_timeout(abcdk_broker_node_t *node, time_t timeout)
{
    int chk;

    assert(node != NULL);

    if (abcdk_atomic_load(&node->stable) != 2)
        return -1;

    chk = abcdk_comm_set_timeout(node->comm, timeout);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_broker_get_sockname(abcdk_broker_node_t *node, abcdk_sockaddr_t *addr)
{
    int chk;

    assert(node != NULL);

    if (abcdk_atomic_load(&node->stable) != 2)
        return -1;

    chk = abcdk_comm_get_sockname(node->comm, addr);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_broker_get_peername(abcdk_broker_node_t *node, abcdk_sockaddr_t *addr)
{
    int chk;

    assert(node != NULL);

    if (abcdk_atomic_load(&node->stable) != 2)
        return -1;

    chk = abcdk_comm_get_peername(node->comm, addr);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_broker_post(abcdk_broker_node_t *node, abcdk_comm_msg_t *msg)
{
    abcdk_tree_t *msg_rsp;
    abcdk_comm_msg_t *msg_rsp_p;
    int chk;

    assert(node != NULL && msg != NULL);

    if (!abcdk_atomic_load(&node->stable))
        goto final_error;

    chk = _abcdk_broker_out_push(node, msg);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    abcdk_comm_msg_unref(&msg);

    return -1;
}

int abcdk_broker_listen(SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_broker_message_cb message_cb, void *opaque)
{
    abcdk_broker_node_t *node;
    int chk;

    assert(addr != NULL && message_cb != NULL);

    node = _abcdk_broker_node_alloc();
    if (!node)
        return -1;

    node->message_cb = message_cb;
    node->opaque = opaque;

    chk = abcdk_comm_listen(ssl_ctx, addr, _abcdk_broker_event_cb, node);
    if (chk == 0)
        return 0;

    abcdk_broker_node_unref(&node);

    return -1;
}

abcdk_broker_node_t *abcdk_broker_connect(SSL_CTX *ssl_ctx,abcdk_sockaddr_t *addr, abcdk_broker_message_cb message_cb, void *opaque)
{
    abcdk_broker_node_t *node;
    abcdk_broker_node_t *node_p;
    int chk;

    assert(addr != NULL && message_cb != NULL);

    node = _abcdk_broker_node_alloc();
    if (!node)
        return NULL;

    /*增加对象引用计数，用于应用层使用。*/
    node_p = abcdk_broker_node_refer(node);

    node->message_cb = message_cb;
    node->opaque = opaque;

    chk = abcdk_comm_connect(ssl_ctx, addr, _abcdk_broker_event_cb, node);
    if (chk == 0)
        return node_p;

    abcdk_broker_node_unref(&node);
    abcdk_broker_node_unref(&node_p);

    return NULL;
}
