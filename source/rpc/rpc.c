/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/rpc/rpc.h"

/*
 * -----------------------------------------------------------------
 * |Message Data                                                   |
 * -----------------------------------------------------------------
 * |Length  |Protocol |Number  |Reserve |Flag    |Reserve |Cargo   |
 * |4 Bytes |4 Bytes  |8 Bytes |1 Bytes |1 Bytes |2 Bytes |N Bytes |
 * -----------------------------------------------------------------
*/

/** 数据包最大长度。*/
#define ABCDK_RPC_MAX_SIZE ((1 << 23) - 1)

/** 数据包头部长度(4+4+8+1+1+2)。*/
#define ABCDK_RPC_HDR_SIZE (20)

/** 默认的数据协议。*/
#define ABCDK_RPC_PROTOCOL 1234567890

/** 应答标志。*/
#define ABCDK_RPC_FLAG_RSP 0x01

/** 简单通信节点。*/
typedef struct _abcdk_rpc_node
{
    /** 魔法数。*/
    uint32_t magic;
#define ABCDK_RPC_MAGIC 123456789

    /**通讯环境指针。*/
    abcdk_comm_t *ctx;

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
    abcdk_rpc_callback_t *callback;
    abcdk_rpc_callback_t cb_cp;

    /** 数据包协议。*/
    uint32_t protocol;

    /** 输入消息缓存。*/
    abcdk_message_t *in_buffer;

    /** 应答服务员。*/
    abcdk_waiter_t *rsp_waiter;

} abcdk_rpc_node_t;


uint64_t _abcdk_rpc_make_mid()
{
    static volatile uint64_t mid = 1;

    return abcdk_atomic_fetch_and_add(&mid, 1);
}

int _abcdk_rpc_post(abcdk_comm_node_t *node, const void *cargo,size_t len, uint64_t num, uint8_t flag)
{
    abcdk_rpc_node_t *rpc_p = NULL;
    abcdk_object_t *msg = NULL;
    int chk;

    rpc_p = (abcdk_rpc_node_t *)abcdk_comm_get_extend0(node);

    msg = abcdk_object_alloc2(4 + 4 + 8 + 1 + 1 + 2 + len);
    if (!msg)
        return -1;
#if 0
    ABCDK_PTR2U32(msg->pptrs[0], 0) = abcdk_endian_h_to_b32(msg->sizes[0]);
    ABCDK_PTR2U32(msg->pptrs[0], 4) = abcdk_endian_h_to_b32(rpc_p->protocol);
    ABCDK_PTR2U64(msg->pptrs[0], 8) = abcdk_endian_h_to_b64(num);
    ABCDK_PTR2U8(msg->pptrs[0], 17) = flag;
#else
    abcdk_bloom_write_number(msg->pptrs[0], 20, 0, 32, msg->sizes[0]);
    abcdk_bloom_write_number(msg->pptrs[0], 20, 32, 32, rpc_p->protocol);
    abcdk_bloom_write_number(msg->pptrs[0], 20, 64, 64, num);
    abcdk_bloom_write_number(msg->pptrs[0], 20, 128, 8, 0);
    abcdk_bloom_write_number(msg->pptrs[0], 20, 136, 8, flag);
    abcdk_bloom_write_number(msg->pptrs[0], 20, 144, 16, 0);
#endif
    memcpy(ABCDK_PTR2VPTR(msg->pptrs[0], ABCDK_RPC_HDR_SIZE), cargo, len);

    chk = abcdk_comm_post(node, msg);
    if (chk == 0)
        return 0;

    /*删除未投递成功的消息。*/
    abcdk_object_unref(&msg);
    return -1;
}

abcdk_message_t *_abcdk_rpc_extrac_cargo(abcdk_message_t *msg)
{
    abcdk_message_t *cargo;
    void *msg_ptr;
    size_t msg_len;
    void *cargo_ptr;
    size_t cargo_len;
    int chk;

    msg_ptr = abcdk_message_data(msg);
    msg_len = abcdk_message_size(msg);

    cargo = abcdk_message_alloc(msg_len - ABCDK_RPC_HDR_SIZE);
    if(!cargo)
        return NULL;

    cargo_ptr = abcdk_message_data(cargo);
    cargo_len = abcdk_message_size(cargo);

    memcpy(cargo_ptr, ABCDK_PTR2VPTR(msg_ptr, ABCDK_RPC_HDR_SIZE), cargo_len);

    return cargo;
}

void _abcdk_rpc_node_destroy_cb(abcdk_object_t *obj, void *opaque)
{
    abcdk_rpc_node_t *rpc_p = NULL;

    rpc_p = (abcdk_rpc_node_t *)obj->pptrs[0];

    abcdk_message_unref(&rpc_p->in_buffer);
    abcdk_waiter_free(&rpc_p->rsp_waiter);
}

abcdk_comm_node_t *abcdk_rpc_alloc(abcdk_comm_t *ctx, uint32_t protocol,size_t userdata)
{
    abcdk_comm_node_t *node = NULL;
    abcdk_rpc_node_t *rpc_p = NULL;
    abcdk_object_t *extend_p = NULL;

    assert(ctx != NULL);
    
    node = abcdk_comm_alloc(ctx,sizeof(abcdk_rpc_node_t),userdata);
    if(!node)
        return NULL;

    extend_p = abcdk_comm_extend(node);
    rpc_p = (abcdk_rpc_node_t *)extend_p->pptrs[0];
    
    /*绑定扩展数据析构函数。*/
    abcdk_object_atfree(extend_p,_abcdk_rpc_node_destroy_cb,NULL);
    abcdk_object_unref(&extend_p);

    rpc_p->magic = ABCDK_RPC_MAGIC;
    rpc_p->ctx = ctx;
    rpc_p->status = ABCDK_RPC_STATUS_BROKEN;
    rpc_p->protocol = protocol;
    rpc_p->in_buffer = NULL;
    rpc_p->rsp_waiter = abcdk_waiter_alloc();
    if(!rpc_p->rsp_waiter)
        goto final_error;

    return node;

final_error:

    abcdk_comm_unref(&node);

    return NULL;
}

void _abcdk_rpc_queue_msg_destroy_cb(const void *msg)
{
    abcdk_message_t *msg_p = (abcdk_message_t *)msg;

    abcdk_message_unref(&msg_p);
}

int abcdk_rpc_request(abcdk_comm_node_t *node, const void *data, size_t len, abcdk_message_t **rsp, time_t timeout)
{
    abcdk_rpc_node_t *rpc_p = NULL;
    abcdk_queue_t *rsp_queue = NULL;
    abcdk_message_t *rsp_msg = NULL;
    uint64_t mid;
    int chk;

    assert(node != NULL && data != NULL && len > 0);
    assert(len <= ABCDK_RPC_MAX_SIZE - ABCDK_RPC_HDR_SIZE);
    ABCDK_ASSERT(rsp == NULL || (rsp != NULL && timeout > 0), "必须指定应答等待时长。");

    rpc_p = (abcdk_rpc_node_t *)abcdk_comm_get_extend0(node);
    ABCDK_ASSERT(rpc_p != NULL && rpc_p->magic == ABCDK_RPC_MAGIC,"未通过rpc接口建立连接，不能调此接口。");

    if(abcdk_atomic_load(&rpc_p->status) == ABCDK_RPC_STATUS_BROKEN)
        return -2;

    mid = _abcdk_rpc_make_mid();

    /*需要应答时，需先创建应答等待对象。*/
    if (rsp)
    {
        rsp_queue = abcdk_queue_alloc(_abcdk_rpc_queue_msg_destroy_cb);
        if(!rsp_queue)
            return -1;

        chk = abcdk_waiter_request(rpc_p->rsp_waiter, mid,rsp_queue);
        if (chk != 0)
        {
            abcdk_queue_free(&rsp_queue);
            return -1;
        }
    }

    /*发送请求(仅向输出队列注册事件和消息)。*/
    chk = _abcdk_rpc_post(node, data, len, mid, 0);
    if (chk != 0)
        return -1;

    /*不需要应答直接返回。*/
    if (!rsp)
        return 0;

    rsp_queue = abcdk_waiter_wait(rpc_p->rsp_waiter, mid, 1, timeout*1000);
    assert(rsp_queue != NULL);

    rsp_msg = (abcdk_message_t*)abcdk_queue_pop(rsp_queue, 1);
    abcdk_queue_free(&rsp_queue);

    if (!rsp_msg)
        return -2;

    /*提取应答货物(应用层数据包)。*/
    *rsp = _abcdk_rpc_extrac_cargo(rsp_msg);
    abcdk_message_unref(&rsp_msg);

    return 0;
}

int abcdk_rpc_response(abcdk_comm_node_t *node,uint64_t mid, const void *data, size_t len)
{
    abcdk_rpc_node_t *rpc_p = NULL;
    int chk;

    assert(node != NULL && data != NULL && len > 0);
    assert(len <= ABCDK_RPC_MAX_SIZE - ABCDK_RPC_HDR_SIZE);

    rpc_p = (abcdk_rpc_node_t *)abcdk_comm_get_extend0(node);
    ABCDK_ASSERT(rpc_p != NULL && rpc_p->magic == ABCDK_RPC_MAGIC,"未通过rpc接口建立连接，不能调此接口。");

    if(abcdk_atomic_load(&rpc_p->status) == ABCDK_RPC_STATUS_BROKEN)
        return -2;

    chk = _abcdk_rpc_post(node, data, len, mid, ABCDK_RPC_FLAG_RSP);
    if (chk != 0)
        return -1;  

    return 0;
}

void _abcdk_rpc_prepare_cb(abcdk_comm_node_t **node, abcdk_comm_node_t *listen)
{
    abcdk_rpc_node_t *rpc_listen_p = NULL;
    abcdk_rpc_node_t *rpc_p = NULL;
    abcdk_object_t *append_p = NULL;
    abcdk_comm_node_t *node_p = NULL;

    rpc_listen_p = (abcdk_rpc_node_t *)abcdk_comm_get_extend0(listen);

    /*如果未指定准备函数，则准备基本的节点环境。*/
    if(rpc_listen_p->callback->prepare_cb)
        rpc_listen_p->callback->prepare_cb(&node_p,listen);
    else 
        node_p = abcdk_rpc_alloc(rpc_listen_p->ctx,rpc_listen_p->protocol,0);

    if(!node_p)
        return;

    rpc_p = (abcdk_rpc_node_t *)abcdk_comm_get_extend0(node_p);
    ABCDK_ASSERT(rpc_p != NULL && rpc_p->magic == ABCDK_RPC_MAGIC,"未通过rpc接口建立连接，不能调此接口。");

    /*复制指针。*/
    rpc_p->callback = rpc_listen_p->callback;

    /*配置参数。*/
    rpc_p->status = ABCDK_RPC_STATUS_SYNC;

    /*准备完毕，返回。*/
    *node = node_p;
}

void _abcdk_rpc_event_accept(abcdk_comm_node_t *node, int *result)
{
    abcdk_rpc_node_t *rpc_p = NULL;

    rpc_p = (abcdk_rpc_node_t *)abcdk_comm_get_extend0(node);

    if(rpc_p->callback->accept_cb)
        rpc_p->callback->accept_cb(node,result);
    else
        *result = 0;
}

void _abcdk_rpc_event_connect(abcdk_comm_node_t *node)
{
    abcdk_rpc_node_t *rpc_p = NULL;
    SSL *ssl_p = NULL;
    int chk;

    rpc_p = (abcdk_rpc_node_t *)abcdk_comm_get_extend0(node);

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

int _abcdk_rpc_msg_unpack(void *opaque, abcdk_message_t *msg)
{   
    abcdk_rpc_node_t *rpc_p = NULL;
    uint32_t len;
    uint32_t pro;
    void *msg_ptr;
    size_t msg_len;
    size_t msg_off;

    rpc_p = (abcdk_rpc_node_t *)opaque;
    
    msg_ptr = abcdk_message_data(msg);
    msg_len = abcdk_message_size(msg);
    msg_off = abcdk_message_offset(msg);

    if (msg_off < ABCDK_RPC_HDR_SIZE)
        return 0;

#if 0
    len = abcdk_endian_b_to_h32(ABCDK_PTR2U32(msg_ptr, 0));
    pro = abcdk_endian_b_to_h32(ABCDK_PTR2U32(msg_ptr, 4));
#else
    len = abcdk_bloom_read_number((uint8_t*)msg_ptr,20,0,32);
    pro = abcdk_bloom_read_number((uint8_t*)msg_ptr,20,32,32);
#endif

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
        abcdk_message_expand(msg, ABCDK_MIN(524288, len - msg_len));
        return 0;
    }

    return 1;
}

void _abcdk_rpc_request_cb(abcdk_comm_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_rpc_node_t *rpc_p = NULL;
    abcdk_message_t *msg_p = NULL;
    void *msg_ptr;
    size_t msg_len;
    uint64_t mid;
    uint8_t flag;
    void *cargo_ptr;
    size_t cargo_len;
    int chk;

    rpc_p = (abcdk_rpc_node_t *)abcdk_comm_get_extend0(node);

    /*准备接收数的缓存。*/
    if (!rpc_p->in_buffer)
    {
        rpc_p->in_buffer = abcdk_message_alloc(ABCDK_RPC_HDR_SIZE);
        if (!rpc_p->in_buffer)
        {
            abcdk_comm_set_timeout(node, 1);
            return;
        }

        /*设置消息协议。*/
        abcdk_message_protocol_t prot = {rpc_p,_abcdk_rpc_msg_unpack};
        abcdk_message_protocol_set(rpc_p->in_buffer, &prot);
    }
    
    chk = abcdk_message_recv(rpc_p->in_buffer,data,size,remain);
    if (chk < 0)
    {
        abcdk_comm_set_timeout(node, 1);
        return;
    }
    else if (chk == 0)
    {
        return;
    }

    /*托管缓存。*/
    msg_p = rpc_p->in_buffer;
    /*缓存已经被托管，这里不能再继续使用了。*/
    rpc_p->in_buffer = NULL;

    /*处理接收到的数据。*/
    msg_ptr = abcdk_message_data(msg_p);
    msg_len = abcdk_message_size(msg_p);

#if 0
    mid = abcdk_endian_b_to_h64(ABCDK_PTR2U64(msg_ptr, 8));
    flag = ABCDK_PTR2U8(msg_ptr, 17);
#else 
    mid = abcdk_bloom_read_number((uint8_t*)msg_ptr,20,64,64);
    flag = abcdk_bloom_read_number((uint8_t*)msg_ptr,20,136,8);
#endif

    cargo_ptr = ABCDK_PTR2VPTR(msg_ptr, ABCDK_RPC_HDR_SIZE);
    cargo_len = msg_len - ABCDK_RPC_HDR_SIZE;

    /*检测是请求还是应答。*/
    if (flag & ABCDK_RPC_FLAG_RSP)
    {
        /*等待应答失败时，直接删除消息。*/
        chk = abcdk_waiter_response(rpc_p->rsp_waiter, mid,msg_p);
        if(chk != 0)
            abcdk_message_unref(&msg_p);
    }
    else
    {
        /*通知应用层，数据到达。*/
        rpc_p->callback->request_cb(node,mid,cargo_ptr,cargo_len);
        /*删除请求数据。*/
        abcdk_message_unref(&msg_p);
    }
}

void _abcdk_rpc_event_input(abcdk_comm_node_t *node)
{
    return;
}

void _abcdk_rpc_event_output(abcdk_comm_node_t *node)
{
    return;
}

void _abcdk_rpc_event_close(abcdk_comm_node_t *node)
{
    abcdk_rpc_node_t *rpc_p = NULL;
    char sockname_str[NAME_MAX] = {0};
    char peername_str[NAME_MAX] = {0};

    rpc_p = (abcdk_rpc_node_t *)abcdk_comm_get_extend0(node);

    if (rpc_p)
    {
        /*标记坏了。*/
        abcdk_atomic_store(&rpc_p->status, ABCDK_RPC_STATUS_BROKEN);

        /*通知所有在这个线路上等待应答的请求，连接已经关闭。*/
        abcdk_waiter_cancel(rpc_p->rsp_waiter);

        /*通知连接已断开。*/
        if(rpc_p->callback->close_cb)
            rpc_p->callback->close_cb(node);
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
    abcdk_rpc_node_t *rpc_p = NULL;
    int chk;

    assert(node != NULL && addr != NULL && cb != NULL);
    ABCDK_ASSERT(cb->request_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");
  
    rpc_p = (abcdk_rpc_node_t *)abcdk_comm_get_extend0(node);
    ABCDK_ASSERT(rpc_p != NULL && rpc_p->magic == ABCDK_RPC_MAGIC,"未通过rpc接口建立连接，不能调此接口。");
    
    /*初始化状态。*/
    rpc_p->status = ABCDK_RPC_STATUS_SYNC;
    rpc_p->cb_cp = *cb;
    rpc_p->callback = &rpc_p->cb_cp;

    abcdk_comm_callback_t fcb = {_abcdk_rpc_prepare_cb,_abcdk_rpc_event_cb,_abcdk_rpc_request_cb};
    chk = abcdk_comm_listen(node, ssl_ctx, addr, &fcb);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    return -1;
}

int abcdk_rpc_connect(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_rpc_callback_t *cb)
{
    abcdk_rpc_node_t *rpc_p = NULL;
    int chk;

    assert(node != NULL && addr != NULL && cb != NULL);
    ABCDK_ASSERT(cb->request_cb != NULL,"未绑定通知回调函数，通讯对象无法正常工作。");

    rpc_p = (abcdk_rpc_node_t *)abcdk_comm_get_extend0(node);
    ABCDK_ASSERT(rpc_p != NULL && rpc_p->magic == ABCDK_RPC_MAGIC,"未通过rpc接口建立连接，不能调此接口。");
    
    /*初始化状态。*/
    rpc_p->status = ABCDK_RPC_STATUS_SYNC;
    rpc_p->cb_cp = *cb;
    rpc_p->callback = &rpc_p->cb_cp;

    abcdk_comm_callback_t fcb = {_abcdk_rpc_prepare_cb,_abcdk_rpc_event_cb,_abcdk_rpc_request_cb};
    chk = abcdk_comm_connect(node, ssl_ctx, addr, &fcb);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    return -1;
}