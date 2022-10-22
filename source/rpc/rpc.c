/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/rpc/rpc.h"

/*
 * --------------------------------------------------------
 * |Message Data                                          |
 * --------------------------------------------------------
 * |Length  |Protocol |Number  |Flag    |Reserve |Cargo   |
 * |4 Bytes |4 Bytes  |8 Bytes |1 Bytes |3 Bytes |N Bytes |
 * --------------------------------------------------------
*/

/** 数据包最大长度。*/
#define ABCDK_RPC_MAX_SIZE ((1 << 23) - 1)

/** 数据包头部长度(4+4+8+1+3)。*/
#define ABCDK_RPC_HDR_SIZE (20)

/** 默认的数据协议。*/
#define ABCDK_RPC_PROTOCOL 1234567890

/** 应答标志。*/
#define ABCDK_RPC_FLAG_RSP 0x01

/** 简单通信节点。*/
typedef struct _abcdk_rpc
{
    /** 魔法数。*/
    uint32_t magic;
#define ABCDK_RPC_MAGIC 123456789

    /** 
     * 状态。
     * 
     * 0：断开(关闭)。
     * 1：连接中(监听中)。
     * 2：已连接。
    */
    volatile int status;
#define ABCDK_RPC_STATUS_BROKEN 0
#define ABCDK_RPC_STATUS_SYNC 1
#define ABCDK_RPC_STATUS_STABLE 2

    /** 通知回调函数。*/
    abcdk_rpc_callback_t callback;

    /** 数据包协议。*/
    uint32_t protocol;

    /** 输入消息缓存。*/
    abcdk_comm_message_t *in_buffer;

    /** 输出消息缓存。*/
    abcdk_comm_message_t *out_buffer;

    /** 输出消息队列。*/
    abcdk_comm_queue_t *out_queue;

    /** 应答服务员。*/
    abcdk_comm_waiter_t *rsp_waiter;

} abcdk_rpc_t;

void _abcdk_rpc_free(abcdk_rpc_t **rpc)
{
    abcdk_rpc_t *rpc_p = NULL;

    if(!rpc || !*rpc)
        return;

    rpc_p = *rpc;
    *rpc = NULL;

    abcdk_comm_message_unref(&rpc_p->in_buffer);
    abcdk_comm_message_unref(&rpc_p->out_buffer);
    abcdk_comm_queue_free(&rpc_p->out_queue);
    abcdk_comm_waiter_free(&rpc_p->rsp_waiter);
    abcdk_heap_free(rpc_p);
}

abcdk_rpc_t *_abcdk_rpc_alloc()
{
    abcdk_rpc_t *rpc = NULL;

    rpc = (abcdk_rpc_t *)abcdk_heap_alloc(sizeof(abcdk_rpc_t));
    if (!rpc)
        return NULL;

    rpc->magic = ABCDK_RPC_MAGIC;
    rpc->status = ABCDK_RPC_STATUS_BROKEN;
    rpc->protocol = ABCDK_RPC_PROTOCOL;
    rpc->in_buffer = NULL;
    rpc->out_buffer = NULL;
    rpc->out_queue = abcdk_comm_queue_alloc();
    rpc->rsp_waiter = abcdk_comm_waiter_alloc();

    return rpc;
}

void _abcdk_rpc_destroy_cb(abcdk_object_t *alloc, void *opaque)
{
    abcdk_rpc_t *rpc_p = NULL;

    if (!alloc->pptrs[0])
        return;

    rpc_p = (abcdk_rpc_t *)alloc->pptrs[0];
    alloc->pptrs[0] = NULL;

    _abcdk_rpc_free(&rpc_p);
}

uint64_t _abcdk_rpc_make_mid()
{
    static volatile uint64_t mid = 1;

    return abcdk_atomic_fetch_and_add(&mid, 1);
}

int _abcdk_rpc_post(abcdk_comm_node_t *node, const void *cargo,size_t len, uint64_t num, uint8_t flag)
{
    abcdk_rpc_t *rpc_p = NULL;
    abcdk_comm_message_t *msg = NULL;
    void *msg_ptr;
    size_t msg_len;
    int chk;

    rpc_p = (abcdk_rpc_t *)abcdk_comm_get_append(node);

    msg = abcdk_comm_message_alloc(4 + 4 + 8 + 1 + 3 + len);
    if (!msg)
        goto final_error;

    msg_ptr = abcdk_comm_message_data(msg);
    msg_len = abcdk_comm_message_size(msg);

    ABCDK_PTR2U32(msg_ptr, 0) = abcdk_endian_h_to_b32(msg_len);
    ABCDK_PTR2U32(msg_ptr, 4) = abcdk_endian_h_to_b32(rpc_p->protocol);
    ABCDK_PTR2U64(msg_ptr, 8) = abcdk_endian_h_to_b64(num);
    ABCDK_PTR2U8(msg_ptr, 17) = flag;
    memcpy(ABCDK_PTR2VPTR(msg_ptr, ABCDK_RPC_HDR_SIZE), cargo, len);

    chk = abcdk_comm_queue_push(rpc_p->out_queue, msg, 0);
    if (chk != 0)
        goto final_error;

    if (abcdk_atomic_load(&rpc_p->status) == ABCDK_RPC_STATUS_STABLE)
        abcdk_comm_send_watch(node);

    return 0;

final_error:

    abcdk_comm_message_unref(&msg);

    return -1;
}

abcdk_comm_message_t *_abcdk_rpc_extrac_cargo(abcdk_comm_message_t *msg)
{
    abcdk_comm_message_t *cargo;
    void *msg_ptr;
    size_t msg_len;
    void *cargo_ptr;
    size_t cargo_len;
    int chk;

    msg_ptr = abcdk_comm_message_data(msg);
    msg_len = abcdk_comm_message_size(msg);

    cargo = abcdk_comm_message_alloc(msg_len - ABCDK_RPC_HDR_SIZE);
    if(!cargo)
        return NULL;

    cargo_ptr = abcdk_comm_message_data(cargo);
    cargo_len = abcdk_comm_message_size(cargo);

    memcpy(cargo_ptr, ABCDK_PTR2VPTR(msg_ptr, ABCDK_RPC_HDR_SIZE), cargo_len);

    return cargo;
}

abcdk_comm_node_t *abcdk_rpc_alloc(abcdk_comm_t *ctx, uint32_t protocol)
{
    abcdk_comm_node_t *node = NULL;
    abcdk_rpc_t *rpc = NULL;
    abcdk_object_t *append_p = NULL;

    assert(ctx != NULL);
    
    node = abcdk_comm_alloc(ctx);
    if(!node)
        return NULL;

    rpc = _abcdk_rpc_alloc();
    if(!rpc)
        goto final_error;

    /*绑定通讯协议。*/
    rpc->protocol = protocol;
    
    append_p = abcdk_comm_append(node);
    append_p->pptrs[0] = (uint8_t*)rpc;
    abcdk_object_atfree(append_p,_abcdk_rpc_destroy_cb,NULL);
    abcdk_object_unref(&append_p);

    return node;

final_error:

    abcdk_comm_unref(&node);

    return NULL;
}

int abcdk_rpc_state(abcdk_comm_node_t *node)
{
    abcdk_rpc_t *rpc_p = NULL;

    assert(node != NULL);

    rpc_p = (abcdk_rpc_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(rpc_p != NULL && rpc_p->magic == ABCDK_RPC_MAGIC,"未通过rpc接口建立连接，不能调此接口。");
    
    if (!abcdk_atomic_load(&rpc_p->status))
        return -1;

    return 0;
}

int abcdk_rpc_request(abcdk_comm_node_t *node, const void *data, size_t len, abcdk_comm_message_t **rsp, time_t timeout)
{
    abcdk_rpc_t *rpc_p = NULL;
    abcdk_comm_queue_t *rsp_queue = NULL;
    abcdk_comm_message_t *rsp_msg = NULL;
    uint64_t mid;
    int chk;

    assert(node != NULL && data != NULL && len > 0);
    assert(len <= ABCDK_RPC_MAX_SIZE - ABCDK_RPC_HDR_SIZE);
    ABCDK_ASSERT(rsp == NULL || (rsp != NULL && timeout > 0), "必须指定应答等待时长。");

    rpc_p = (abcdk_rpc_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(rpc_p != NULL && rpc_p->magic == ABCDK_RPC_MAGIC,"未通过rpc接口建立连接，不能调此接口。");

    if(!abcdk_atomic_load(&rpc_p->status))
        return -2;

    mid = _abcdk_rpc_make_mid();

    if (rsp)
        abcdk_comm_waiter_request2(rpc_p->rsp_waiter, &mid);

    /*发送请求(仅向输出队列注册事件和消息)。*/
    chk = _abcdk_rpc_post(node, data, len, mid, 0);
    if (chk != 0)
        return -1;

    /*不需要应答直接返回。*/
    if (!rsp)
        return 0;

    rsp_queue = abcdk_comm_waiter_wait2(rpc_p->rsp_waiter, &mid, 1, timeout*1000);
    if (!rsp_queue)
        return -1;

    rsp_msg = abcdk_comm_queue_pop(rsp_queue, 1);
    abcdk_comm_queue_free(&rsp_queue);

    if (!rsp_msg)
        return -2;

    /*提取应答货物(应用层数据包)。*/
    *rsp = _abcdk_rpc_extrac_cargo(rsp_msg);
    abcdk_comm_message_unref(&rsp_msg);

    return 0;
}

int abcdk_rpc_response(abcdk_comm_node_t *node,uint64_t mid, const void *data, size_t len)
{
    abcdk_rpc_t *rpc_p = NULL;
    int chk;

    assert(node != NULL && data != NULL && len > 0);
    assert(len <= ABCDK_RPC_MAX_SIZE - ABCDK_RPC_HDR_SIZE);

    rpc_p = (abcdk_rpc_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(rpc_p != NULL && rpc_p->magic == ABCDK_RPC_MAGIC,"未通过rpc接口建立连接，不能调此接口。");

    if(!abcdk_atomic_load(&rpc_p->status))
        return -2;

    chk = _abcdk_rpc_post(node, data, len, mid, ABCDK_RPC_FLAG_RSP);
    if (chk != 0)
        return -1;  

    return 0;
}

void _abcdk_rpc_prepare_cb(abcdk_comm_node_t *node, abcdk_comm_node_t *listen)
{
    abcdk_rpc_t *listen_rpc_p = NULL;
    abcdk_object_t *append_p = NULL;
    abcdk_rpc_t *rpc_p = NULL;

    listen_rpc_p = (abcdk_rpc_t *)abcdk_comm_get_append(listen);

    rpc_p = _abcdk_rpc_alloc();
    if (!rpc_p)
        return;

    append_p = abcdk_comm_append(node);
    append_p->pptrs[0] = (uint8_t*)rpc_p;
    abcdk_object_atfree(append_p,_abcdk_rpc_destroy_cb,NULL);
    abcdk_object_unref(&append_p);

    /*复制通讯协议。*/
    rpc_p->protocol = listen_rpc_p->protocol;
    /*复制请求回调函数指针。*/
    rpc_p->callback = listen_rpc_p->callback;
    /*复制监听的用户环境指针。*/
    abcdk_comm_set_userdata(node,abcdk_comm_get_userdata(listen));

}

void _abcdk_rpc_event_accept(abcdk_comm_node_t *node, int *result)
{
    abcdk_rpc_t *rpc_p = NULL;

    rpc_p = (abcdk_rpc_t *)abcdk_comm_get_append(node);

    if(rpc_p->callback.accept_cb)
        rpc_p->callback.accept_cb(node,result);
    else
        *result = 0;
}

void _abcdk_rpc_event_connect(abcdk_comm_node_t *node)
{
    abcdk_rpc_t *rpc_p = NULL;
    SSL *ssl_p = NULL;
    int chk;

    rpc_p = (abcdk_rpc_t *)abcdk_comm_get_append(node);

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
    abcdk_atomic_store(&rpc_p->status, ABCDK_RPC_STATUS_STABLE);
    /*已连接到远端，注册读写事件。*/
    abcdk_comm_recv_watch(node);
    abcdk_comm_send_watch(node);
}

int _abcdk_rpc_msg_unpack(void *opaque, abcdk_comm_message_t *msg)
{   
    abcdk_rpc_t *rpc_p = NULL;
    uint32_t len;
    uint32_t pro;
    void *msg_ptr;
    size_t msg_len;
    size_t msg_off;

    rpc_p = (abcdk_rpc_t *)opaque;
    
    msg_ptr = abcdk_comm_message_data(msg);
    msg_len = abcdk_comm_message_size(msg);
    msg_off = abcdk_comm_message_offset(msg);

    if (msg_off < ABCDK_RPC_HDR_SIZE)
        return 0;

    len = abcdk_endian_b_to_h32(ABCDK_PTR2U32(msg_ptr, 0));
    pro = abcdk_endian_b_to_h32(ABCDK_PTR2U32(msg_ptr, 4));

    /*数据包大小必须符合要求。*/
    if (len > ABCDK_RPC_MAX_SIZE || len < ABCDK_RPC_HDR_SIZE)
        return -1;

    /*仅支持相同的协议。*/
    if(pro != rpc_p->protocol)
        return -1;

    /*如果未收完，继续收。*/
    if (msg_off < len)
    {
        /*增量扩展内存。*/
        abcdk_comm_message_expand(msg, ABCDK_MIN(524288, len - msg_len));
        return 0;
    }

    return 1;
}

void _abcdk_rpc_event_input(abcdk_comm_node_t *node)
{
    abcdk_rpc_t *rpc_p = NULL;
    abcdk_comm_message_t *msg = NULL;
    void *msg_ptr;
    size_t msg_len;
    uint64_t mid;
    uint8_t flag;
    void *cargo_ptr;
    size_t cargo_len;
    int chk;

    rpc_p = (abcdk_rpc_t *)abcdk_comm_get_append(node);

    /*准备接收数的缓存。*/
    if (!rpc_p->in_buffer)
    {
        rpc_p->in_buffer = abcdk_comm_message_alloc(ABCDK_RPC_HDR_SIZE);
        if (!rpc_p->in_buffer)
        {
            abcdk_comm_set_timeout(node, 1);
            return;
        }

        /*设置消息协议。*/
        abcdk_comm_message_protocol_t prot = {rpc_p,_abcdk_rpc_msg_unpack};
        abcdk_comm_message_protocol_set(rpc_p->in_buffer, &prot);
    }
    

    chk = abcdk_comm_message_recv(rpc_p->in_buffer,node);
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
    msg = rpc_p->in_buffer;
    /*缓存已经被托管，这里不能再继续使用了。*/
    rpc_p->in_buffer = NULL;
    
    /*复用链路前要增加引用计数，以防止多线程操作同一个链路在释放回收内存后，造成应用层内存非法访问的异常。*/
    abcdk_comm_refer(node);
    /*复用链路。*/
    abcdk_comm_recv_watch(node);

    /*处理接收到的数据。*/
    msg_ptr = abcdk_comm_message_data(msg);
    msg_len = abcdk_comm_message_size(msg);

    mid = abcdk_endian_b_to_h64(ABCDK_PTR2U64(msg_ptr, 8));
    flag = ABCDK_PTR2U8(msg_ptr, 17);

    cargo_ptr = ABCDK_PTR2VPTR(msg_ptr, ABCDK_RPC_HDR_SIZE);
    cargo_len = msg_len - ABCDK_RPC_HDR_SIZE;

    /*检测是请求还是应答。*/
    if (flag & ABCDK_RPC_FLAG_RSP)
    {
        abcdk_comm_waiter_response2(rpc_p->rsp_waiter, &mid, msg);
    }
    else
    {
        /*通知应用层，数据到达。*/
        rpc_p->callback.request_cb(node,mid,cargo_ptr,cargo_len);

        /*删除请求数据。*/
        abcdk_comm_message_unref(&msg);
    }

    /*减少引用计数。*/
    abcdk_comm_unref(&node);
}

void _abcdk_rpc_event_output(abcdk_comm_node_t *node)
{
    abcdk_rpc_t *rpc_p = NULL;
    int chk;

    rpc_p = (abcdk_rpc_t *)abcdk_comm_get_append(node);

NEXT_MSG:

    /*如果发送缓存是空的，则从待发送队列取出一份。*/
    if (!rpc_p->out_buffer)
    {
        rpc_p->out_buffer = abcdk_comm_queue_pop(rpc_p->out_queue, 1);
        if (!rpc_p->out_buffer)
            return;
    }

    chk = abcdk_comm_message_send(rpc_p->out_buffer,node);
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
    abcdk_comm_message_unref(&rpc_p->out_buffer);
    goto NEXT_MSG;
}

void _abcdk_rpc_event_close(abcdk_comm_node_t *node)
{
    abcdk_rpc_t *rpc_p = NULL;
    char sockname_str[NAME_MAX] = {0};
    char peername_str[NAME_MAX] = {0};

    rpc_p = (abcdk_rpc_t *)abcdk_comm_get_append(node);

    if (rpc_p)
    {
        /*标记坏了。*/
        abcdk_atomic_store(&rpc_p->status, ABCDK_RPC_STATUS_BROKEN);

        /*通知所有在这个线路上等待应答的请求，连接已经关闭。*/
        abcdk_comm_waiter_cancel(rpc_p->rsp_waiter);

        /*通知连接已断开。*/
        if(rpc_p->callback.close_cb)
            rpc_p->callback.close_cb(node);
    }
    else
    {
        /*可能还未完成连接就已经断开了。*/

        abcdk_comm_get_sockaddr_str(node, sockname_str, peername_str);
        fprintf(stderr, "warning: sockname(%s) -> peername(%s) disconnected.\n",sockname_str, peername_str);
    }
}

void _abcdk_rpc_event_cb(abcdk_comm_node_t *node, uint32_t event, int *result)
{
    switch (event)
    {
    case ABCDK_COMM_EVENT_ACCEPT:
        _abcdk_rpc_event_accept(node,result);
        break;
    case ABCDK_COMM_EVENT_CONNECT:
        _abcdk_rpc_event_connect(node);
        break;
    case ABCDK_COMM_EVENT_INPUT:
        _abcdk_rpc_event_input(node);
        break;
    case ABCDK_COMM_EVENT_OUTPUT:
        _abcdk_rpc_event_output(node);
        break;
    case ABCDK_COMM_EVENT_CLOSE:
    case ABCDK_COMM_EVENT_INTERRUPT:
    default:
        _abcdk_rpc_event_close(node);
        break;
    }
}


int abcdk_rpc_listen(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_rpc_callback_t *cb)
{
    abcdk_rpc_t *rpc_p = NULL;
    int chk;

    assert(node != NULL && addr != NULL && cb != NULL);
    ABCDK_ASSERT(cb->request_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");
  
    rpc_p = (abcdk_rpc_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(rpc_p != NULL && rpc_p->magic == ABCDK_RPC_MAGIC,"未通过rpc接口建立连接，不能调此接口。");
    
    /*初始化状态。*/
    rpc_p->status = ABCDK_RPC_STATUS_SYNC;
    rpc_p->callback = *cb;

    abcdk_comm_callback_t fcb = {_abcdk_rpc_prepare_cb,_abcdk_rpc_event_cb};
    chk = abcdk_comm_listen(node, ssl_ctx, addr, &fcb);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    return -1;
}

int abcdk_rpc_connect(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_rpc_callback_t *cb)
{
    abcdk_rpc_t *rpc_p = NULL;
    int chk;

    assert(node != NULL && addr != NULL && cb != NULL);
    ABCDK_ASSERT(cb->request_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");

    rpc_p = (abcdk_rpc_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(rpc_p != NULL && rpc_p->magic == ABCDK_RPC_MAGIC,"未通过rpc接口建立连接，不能调此接口。");
    
    /*初始化状态。*/
    rpc_p->status = ABCDK_RPC_STATUS_SYNC;
    rpc_p->callback = *cb;

    abcdk_comm_callback_t fcb = {_abcdk_rpc_prepare_cb,_abcdk_rpc_event_cb};
    chk = abcdk_comm_connect(node, ssl_ctx, addr, &fcb);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    return -1;
}