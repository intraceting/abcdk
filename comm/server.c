/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-comm/server.h"

/**通信节点。*/
typedef struct _abcdk_comm_svr_node
{
    /*通信链路。*/
    abcdk_comm_node_t *comm;

    /*消息回调。*/
    abcdk_comm_svr_message_cb message_cb;

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

} abcdk_comm_svr_node_t;

void _abcdk_comm_svr_node_free(abcdk_comm_svr_node_t **node)
{
    abcdk_comm_svr_node_t *node_p;

    if (!node || !*node)
        return;

    node_p = *node;

    abcdk_tree_free(&node_p->in_buffer);
    abcdk_tree_free(&node_p->out_buffer);
    abcdk_tree_free(&node_p->out_queue);
    abcdk_mutex_destroy(&node_p->out_locker);
    abcdk_comm_node_unref(&node_p->comm);

    abcdk_heap_free2((void **)node);
}

abcdk_comm_svr_node_t *_abcdk_comm_svr_node_alloc()
{
    abcdk_comm_svr_node_t *node;

    node = abcdk_heap_alloc(sizeof(abcdk_comm_svr_node_t));
    if (!node)
        return NULL;

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

    _abcdk_comm_svr_node_free(&node);

    return NULL;
}

void _abcdk_comm_svr_msg_destroy_cb(abcdk_allocator_t *alloc, void *opaque)
{
    abcdk_comm_msg_t *msg_p = NULL;

    msg_p = (abcdk_comm_msg_t *)alloc->pptrs[0];

    abcdk_comm_msg_free(&msg_p);
}

abcdk_tree_t *_abcdk_comm_svr_msg_alloc(size_t size)
{
    abcdk_tree_t *msg = NULL;

    msg = abcdk_tree_alloc3(0);
    if (!msg)
        return NULL;

    abcdk_allocator_atfree(msg->alloc, _abcdk_comm_svr_msg_destroy_cb, NULL);

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

void _abcdk_comm_svr_accept_event(abcdk_comm_node_t *comm)
{
    abcdk_comm_svr_node_t *node_listen;
    abcdk_comm_svr_node_t *node_accpet;

    node_accpet = _abcdk_comm_svr_node_alloc();
    if (!node_accpet)
    {
        abcdk_comm_set_timeout(comm, 1);
        return;
    }

    node_listen = (abcdk_comm_svr_node_t *)abcdk_comm_get_userdata(comm);
    if (!node_listen)
    {
        abcdk_comm_set_timeout(comm, 1);
        return;
    }

    node_accpet->message_cb = node_listen->message_cb;
    node_accpet->opaque = node_listen->opaque;
    node_accpet->comm = abcdk_comm_node_refer(comm);

    /*通过listen节点建立的连接，初始环境是listen节点的环境指针，现在替换成私有环境指针。*/
    abcdk_comm_set_userdata(comm, node_accpet);

    /*默认30秒超时。*/
    abcdk_comm_set_timeout(comm, 30000);

    /*监听接收事件。*/
    abcdk_comm_read_watch(comm,0);
}

void _abcdk_comm_svr_input_event(abcdk_comm_node_t *comm)
{
    abcdk_comm_svr_node_t *node;
    abcdk_tree_t *msg_req;
    abcdk_tree_t *msg_rsp;
    abcdk_comm_msg_t *msg_req_p;
    abcdk_comm_msg_t *msg_rsp_p;
    int chk;

    node = (abcdk_comm_svr_node_t *)abcdk_comm_get_userdata(comm);
    if (!node)
    {
        abcdk_comm_set_timeout(comm, 1);
        return;
    }

    if (!node->in_buffer)
    {
        node->in_buffer = _abcdk_comm_svr_msg_alloc(64);
        if (!node->in_buffer)
        {
            abcdk_comm_set_timeout(comm, 1);
            return;
        }
    }

    msg_req = node->in_buffer;
    msg_req_p = (abcdk_comm_msg_t *)msg_req->alloc->pptrs[0];

    chk = abcdk_comm_msg_recv(comm, msg_req_p);
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

        /*通知应用层请求消息达到。*/
        node->message_cb(node, msg_req_p, &msg_rsp_p, node->opaque);

        /*删除请求消息。*/
        abcdk_tree_free(&msg_req);

        /*如果有应答则加入到发送队列。*/
        if (msg_rsp_p)
        {
            msg_rsp = _abcdk_comm_svr_msg_alloc(0);
            if (!msg_rsp)
            {
                abcdk_comm_set_timeout(comm, 1);
                return;
            }

            msg_rsp->alloc->pptrs[0] = (uint8_t *)msg_rsp_p;

            abcdk_mutex_lock(&node->out_locker, 1);
            abcdk_tree_insert2(node->out_queue, msg_rsp, 0);
            abcdk_mutex_unlock(&node->out_locker);

            abcdk_comm_write_watch(comm);
        }
    }
}

void _abcdk_comm_svr_output_event(abcdk_comm_node_t *comm)
{
    abcdk_comm_svr_node_t *node;
    abcdk_tree_t *msg_rsp;
    abcdk_comm_msg_t *msg_rsp_p;
    int chk;

    node = (abcdk_comm_svr_node_t *)abcdk_comm_get_userdata(comm);
    if (!node)
    {
        abcdk_comm_set_timeout(comm, 1);
        return;
    }

NEXT_MSG:

    msg_rsp = NULL;
    msg_rsp_p = NULL;

    if (!node->out_buffer)
    {
        abcdk_mutex_lock(&node->out_locker, 1);
        node->out_buffer = abcdk_tree_child(node->out_queue, 1);
        if (node->out_buffer)
            abcdk_tree_unlink(node->out_buffer);
        abcdk_mutex_unlock(&node->out_locker);

        if (!node->out_buffer)
            return;
    }

    msg_rsp = node->out_buffer;
    msg_rsp_p = (abcdk_comm_msg_t *)msg_rsp->alloc->pptrs[0];

    chk = abcdk_comm_msg_send(comm, msg_rsp_p);
    if (chk < 0)
    {
        abcdk_comm_set_timeout(comm, 1);
        return;
    }
    else if (chk > 0)
    {
        /*释放消息缓存，继续发送。*/
        abcdk_tree_free(&node->out_buffer);
        goto NEXT_MSG;
    }

    abcdk_comm_write_watch(comm);
}

void _abcdk_comm_svr_close_event(abcdk_comm_node_t *comm)
{
    abcdk_comm_svr_node_t *node;

    node = (abcdk_comm_svr_node_t *)abcdk_comm_get_userdata(comm);
    if (!node)
        return;

    /*通知应连接关闭。*/
    node->message_cb(node, NULL, NULL, node->opaque);

    _abcdk_comm_svr_node_free(&node);
}

void _abcdk_comm_svr_listen_close_event(abcdk_comm_node_t *comm)
{
    abcdk_comm_svr_node_t *node;

    node = (abcdk_comm_svr_node_t *)abcdk_comm_get_userdata(comm);
    if (!node)
        return;

    _abcdk_comm_svr_node_free(&node);
}

void _abcdk_comm_svr_event_cb(abcdk_comm_node_t *node, uint32_t event)
{
    switch (event)
    {
    case ABCDK_COMM_EVENT_ACCEPT:
    {
        _abcdk_comm_svr_accept_event(node);
    }
    break;
    case ABCDK_COMM_EVENT_INPUT:
    {
        _abcdk_comm_svr_input_event(node);
    }
    break;
    case ABCDK_COMM_EVENT_OUTPUT:
    {
        _abcdk_comm_svr_output_event(node);
    }
    break;
    case ABCDK_COMM_EVENT_CLOSE:
    {
        _abcdk_comm_svr_close_event(node);
    }
    break;
    case ABCDK_COMM_EVENT_LISTEN_CLOSE:
    {
        _abcdk_comm_svr_listen_close_event(node);
    }
    break;
    }
}

int abcdk_comm_svr_set_timeout(abcdk_comm_svr_node_t *node, time_t timeout)
{
    int chk;

    assert(node != NULL);

    chk = abcdk_comm_set_timeout(node->comm,timeout);
    if(chk != 0)
        return -1;

    return 0;
}

int abcdk_comm_svr_get_sockname(abcdk_comm_svr_node_t *node, abcdk_sockaddr_t *addr)
{
    int chk;

    assert(node != NULL);

    chk = abcdk_comm_get_sockname(node->comm,addr);
    if(chk != 0)
        return -1;

    return 0;
}

int abcdk_comm_svr_get_peername(abcdk_comm_svr_node_t *node, abcdk_sockaddr_t *addr)
{
    int chk;

    assert(node != NULL);

    chk = abcdk_comm_get_peername(node->comm,addr);
    if(chk != 0)
        return -1;

    return 0;
}

int abcdk_comm_svr_listen(SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_comm_svr_message_cb message_cb, void *opaque)
{
    abcdk_comm_svr_node_t *node;
    int chk;

    assert(addr != NULL && message_cb != NULL);

    node = _abcdk_comm_svr_node_alloc();
    if (!node)
        return -1;

    node->message_cb = message_cb;
    node->opaque = opaque;

    chk = abcdk_comm_listen(ssl_ctx, addr, _abcdk_comm_svr_event_cb, node);
    if (chk == 0)
        return 0;

    _abcdk_comm_svr_node_free(&node);

    return -1;
}