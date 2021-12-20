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

    /** 消息编号索引。*/
    volatile uint64_t num_index;

    /*通信链路。*/
    abcdk_comm_node_t *comm;

    /*消息回调。*/
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

    /** 应答队列锁。*/
    abcdk_mutex_t rsp_locker;

    /** 应答队列。*/
    abcdk_tree_t *rsp_queue;

} abcdk_broker_node_t;

void _abcdk_broker_node_unref(abcdk_broker_node_t **node)
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
    abcdk_tree_free(&node_p->rsp_queue);
    abcdk_mutex_destroy(&node_p->rsp_locker);
    abcdk_comm_node_unref(&node_p->comm);

    abcdk_heap_free(node_p);

final:

    /*set NULL(0).*/
    *node = NULL;
}

abcdk_broker_node_t *_abcdk_broker_node_refer(abcdk_broker_node_t *src)
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
    node->num_index = 1;
    node->message_cb = NULL;
    node->opaque = NULL;

    node->in_buffer = NULL;
    node->out_buffer = NULL;

    abcdk_mutex_init2(&node->out_locker, 0);
    node->out_queue = abcdk_tree_alloc3(1);
    if (!node->out_queue)
        goto final_error;

    abcdk_mutex_init2(&node->rsp_locker, 0);
    node->rsp_queue = abcdk_tree_alloc3(1);
    if (!node->rsp_queue)
        goto final_error;

    return node;

final_error:

    _abcdk_broker_node_unref(&node);

    return NULL;
}

void _abcdk_broker_msg_destroy_cb(abcdk_allocator_t *alloc, void *opaque)
{
    abcdk_comm_msg_t *msg_p = NULL;

    msg_p = (abcdk_comm_msg_t *)alloc->pptrs[0];

    abcdk_comm_msg_free(&msg_p);
}

abcdk_tree_t *_abcdk_broker_msg_alloc(size_t size)
{
    abcdk_tree_t *msg = NULL;

    /*MSG ptr,MSG number*/
    size_t sizes[] = {0, sizeof(uint64_t)};
    msg = abcdk_tree_alloc2(sizes, 2, 0);
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

    msg_node = _abcdk_broker_msg_alloc2(msg);
    if (!msg_node)
        return -1;

    abcdk_mutex_lock(&node->out_locker, 1);

    abcdk_tree_insert2(node->out_queue, msg_node, 0);

    abcdk_mutex_unlock(&node->out_locker);

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

void _abcdk_broker_rsp_arrived(abcdk_broker_node_t *node, abcdk_tree_t *msg)
{
    abcdk_tree_t *rsp_it;

    assert(abcdk_comm_msg_flag(msg) & ABCDK_COMM_MSG_FLAG_RSP);

    abcdk_mutex_lock(&node->rsp_locker, 1);

    rsp_it = abcdk_tree_child(node->rsp_queue, 1);
    while (rsp_it)
    {
        if (ABCDK_PTR2U64(rsp_it->alloc->pptrs[1], 0) == abcdk_comm_msg_number(msg))
            break;

        rsp_it = abcdk_tree_sibling(rsp_it, 0);
    }

    if (rsp_it)
    {
        /*复制消息指针，并解除原绑定关系。*/
        rsp_it->alloc->pptrs[0] = msg->alloc->pptrs[0];
        msg->alloc->pptrs[0] = NULL;

        abcdk_mutex_signal(&node->rsp_locker, 1);
    }

    abcdk_mutex_unlock(&node->rsp_locker);

    /*删除无效的消息节点。*/
    abcdk_tree_free(&msg);
}

abcdk_comm_msg_t *_abcdk_broker_rsp_wait(abcdk_broker_node_t *node, uint64_t number, time_t timeout)
{
    abcdk_tree_t *rsp_it;
    abcdk_comm_msg_t *msg_p;
    time_t time_end;
    time_t time_span;

    /*找到应答节点。*/
    abcdk_mutex_lock(&node->rsp_locker, 1);

    rsp_it = abcdk_tree_child(node->rsp_queue, 1);
    while (rsp_it)
    {
        if (ABCDK_PTR2U64(rsp_it->alloc->pptrs[1], 0) == number)
            break;
        rsp_it = abcdk_tree_sibling(rsp_it, 0);
    }

    abcdk_mutex_unlock(&node->rsp_locker);

    /*计算过期时间。*/
    time_end = abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 3) + timeout;

    /*等待应答。*/
    abcdk_mutex_lock(&node->rsp_locker, 1);

    while (!rsp_it->alloc->pptrs[0])
    {
        /*计算剩余超时时长。*/
        time_span = time_end - abcdk_time_clock2kind_with(CLOCK_MONOTONIC, 3);
        if (time_span > 0)
            abcdk_mutex_wait(&node->rsp_locker, time_span);
        else
            break;
    }

    /*从应答队列中移除。*/
    abcdk_tree_unlink(rsp_it);

    abcdk_mutex_unlock(&node->rsp_locker);

    if (rsp_it)
    {
        /*复制消息指针，并解除原绑定关系。*/
        msg_p = (abcdk_comm_msg_t *)rsp_it->alloc->pptrs[0];
        rsp_it->alloc->pptrs[0] = NULL;

        /*删除无用的消息节点。*/
        abcdk_tree_free(&rsp_it);
    }

    return msg_p;
}

int _abcdk_broker_rsp_mark(abcdk_broker_node_t *node, uint64_t number)
{
    abcdk_tree_t *rsp_it;
    abcdk_tree_t *msg_node;
    int chk = 0;

    msg_node = _abcdk_broker_msg_alloc(0);
    if (!msg_node)
        return -1;

    /*绑定应答编号。*/
    ABCDK_PTR2U64(rsp_it->alloc->pptrs[1], 0) == number;

    abcdk_mutex_lock(&node->rsp_locker, 1);

    rsp_it = abcdk_tree_child(node->rsp_queue, 1);
    while (rsp_it)
    {
        if (ABCDK_PTR2U64(rsp_it->alloc->pptrs[1], 0) == number)
            break;
        rsp_it = abcdk_tree_sibling(rsp_it, 0);
    }

    /*不能添加重复的应答编号。*/
    if (!rsp_it)
        abcdk_tree_insert2(node->rsp_queue, msg_node, 0);
    else
        chk = -1;

    abcdk_mutex_unlock(&node->rsp_locker);

    if (chk != 0)
        abcdk_tree_free(&msg_node);

    return chk;
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
    node_accpet->comm = abcdk_comm_node_refer(comm);

    /*通过listen节点建立的连接，初始环境是listen节点的环境指针，现在替换成私有环境指针。*/
    abcdk_comm_set_userdata(comm, node_accpet);

    /*默认30秒超时。*/
    abcdk_comm_set_timeout(comm, 30000);

    /*监听接收事件。*/
    abcdk_comm_read_watch(comm, 0);
}

void _abcdk_broker_input_event(abcdk_comm_node_t *comm)
{
    abcdk_broker_node_t *node;
    abcdk_tree_t *msg_in;
    abcdk_tree_t *msg_out;
    abcdk_comm_msg_t *msg_in_p;
    abcdk_comm_msg_t *msg_out_p;
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

        if (abcdk_comm_msg_flag(msg_in_p) & ABCDK_COMM_MSG_FLAG_RSP)
        {
            /*消息答应。*/
            _abcdk_broker_rsp_arrived(node, msg_in);
        }
        else
        {
            /*通知应用层消息达到。*/
            node->message_cb(node, msg_in_p, &msg_out_p, node->opaque);

            /*如果有应答则加入到发送队列。*/
            if (msg_out_p)
            {
                /*复制消息协议、编号，添加应答标志。*/
                abcdk_comm_msg_protocol_set(msg_out_p, abcdk_comm_msg_protocol(msg_in_p));
                abcdk_comm_msg_number_set(msg_out_p, abcdk_comm_msg_number(msg_in_p));
                abcdk_comm_msg_flag_set(msg_out_p, ABCDK_COMM_MSG_FLAG_RSP);

                chk = _abcdk_broker_out_push(node, msg_out_p);
                if (chk != 0)
                    abcdk_comm_msg_free(&msg_out_p);
            }

            /*删除已经处理的消息。*/
            abcdk_tree_free(&msg_in);
        }
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
        /*释放消息缓存，继续发送。*/
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

    /*通知应用层连接已关闭。*/
    node->message_cb(node, NULL, NULL, node->opaque);

    _abcdk_broker_node_unref(&node);
}

void _abcdk_broker_listen_close_event(abcdk_comm_node_t *comm)
{
    abcdk_broker_node_t *node;

    node = (abcdk_broker_node_t *)abcdk_comm_get_userdata(comm);
    if (!node)
        return;

    _abcdk_broker_node_unref(&node);
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

    chk = abcdk_comm_set_timeout(node->comm, timeout);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_broker_get_sockname(abcdk_broker_node_t *node, abcdk_sockaddr_t *addr)
{
    int chk;

    assert(node != NULL);

    chk = abcdk_comm_get_sockname(node->comm, addr);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_broker_get_peername(abcdk_broker_node_t *node, abcdk_sockaddr_t *addr)
{
    int chk;

    assert(node != NULL);

    chk = abcdk_comm_get_peername(node->comm, addr);
    if (chk != 0)
        return -1;

    return 0;
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

    _abcdk_broker_node_unref(&node);

    return -1;
}

int abcdk_broker_transmit(abcdk_broker_node_t *node, abcdk_comm_msg_t *req, abcdk_comm_msg_t **rsp, time_t timeout)
{
    abcdk_tree_t *msg_rsp;
    abcdk_comm_msg_t *msg_rsp_p;
    int chk;

    assert(node != NULL && req != NULL);

    /* 添加协议、编号、标志。*/
    abcdk_comm_msg_protocol_set(req, 1234567890);
    abcdk_comm_msg_number_set(req, abcdk_atomic_fetch_and_add(&node->num_index, 1));
    abcdk_comm_msg_flag_set(req, 0);

    if (rsp)
    {
        /*注册应答。*/
        chk = _abcdk_broker_rsp_mark(node, abcdk_comm_msg_number(req));
        if (chk != 0)
            return -3;
    }

    chk = _abcdk_broker_out_push(node, req);
    if (chk != 0)
    {
        abcdk_comm_msg_free(&req);
        return -2;
    }

    if (rsp)
    {
        /*等待应答。*/
        *rsp = _abcdk_broker_rsp_wait(node, abcdk_comm_msg_number(req), timeout);
        if (*rsp != NULL)
            return 0;
        else
            return -1;
    }

    return 0;
}