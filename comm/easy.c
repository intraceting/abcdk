/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "comm/easy.h"

/*
 * --------------------------------------------------------
 * |Message Data                                          |
 * --------------------------------------------------------
 * |Length  |Protocol |Number  |Flag    |Reserve |Cargo   |
 * |4 Bytes |4 Bytes  |8 Bytes |1 Bytes |3 Bytes |N Bytes |
 * --------------------------------------------------------
*/

/** 数据包头部长度(4+4+8+1+3)。*/
#define ABCDK_COMM_EASY_MD_HDR_SIZE (20)

/** 数据包最大长度。*/
#define ABCDK_COMM_EASY_MD_MAX_SIZE ((256 * 1024 * 1024) - 1)

/** 数据协议。*/
#define ABCDK_COMM_EASY_MD_PROTOCOL 1234567890

/** 应答标志。*/
#define ABCDK_COMM_EASY_MD_FLAG_RSP 0x01

/** 简单通信节点。*/
typedef struct _abcdk_comm_easy
{
    /** 魔法数。*/
    uint32_t magic;
#define ABCDK_COMM_EASY_MAGIC 20220819

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

    /** 请求回调函数指针。*/
    abcdk_comm_easy_request_cb request_cb;

    /** 输入消息缓存。*/
    abcdk_comm_message_t *in_buffer;

    /** 输出消息缓存。*/
    abcdk_comm_message_t *out_buffer;

    /** 输出消息队列。*/
    abcdk_comm_queue_t *out_queue;

    /** 应答服务员。*/
    abcdk_comm_waiter_t *rsp_waiter;

    /** 请求消息线程KEY。*/
    pthread_key_t req_ptkey;

} abcdk_comm_easy_t;

void _abcdk_comm_easy_free(abcdk_comm_easy_t **easy)
{
    abcdk_comm_easy_t *easy_p = NULL;

    if(!easy || !*easy)
        return;

    easy_p = *easy;
    *easy = NULL;


    abcdk_comm_message_unref(&easy_p->in_buffer);
    abcdk_comm_queue_free(&easy_p->out_queue);
    abcdk_comm_waiter_free(&easy_p->rsp_waiter);
    pthread_key_delete(easy_p->req_ptkey);
    abcdk_heap_free(easy_p);
}

abcdk_comm_easy_t *_abcdk_comm_easy_alloc()
{
    abcdk_comm_easy_t *easy = NULL;

    easy = (abcdk_comm_easy_t *)abcdk_heap_alloc(sizeof(abcdk_comm_easy_t));
    if (!easy)
        return NULL;

    easy->magic = ABCDK_COMM_EASY_MAGIC;
    easy->flag = 0;
    easy->status = 1;
    easy->in_buffer = NULL;
    easy->out_buffer = NULL;
    easy->out_queue = abcdk_comm_queue_alloc();
    easy->rsp_waiter = abcdk_comm_waiter_alloc();
    pthread_key_create(&easy->req_ptkey,NULL);

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

uint64_t _abcdk_comm_easy_make_mid()
{
    static volatile uint64_t mid = 1;

    return abcdk_atomic_fetch_and_add(&mid, 1);
}

int _abcdk_comm_easy_post(abcdk_comm_node_t *node, const void *cargo,size_t len, uint64_t num, uint8_t flag)
{
    abcdk_comm_easy_t *easy_p = NULL;
    abcdk_comm_message_t *msg = NULL;
    void *msg_ptr;
    size_t msg_len;
    int chk;

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);

    msg = abcdk_comm_message_alloc(4 + 4 + 8 + 1 + 3 + len);
    if (!msg)
        goto final_error;

    msg_ptr = abcdk_comm_message_data(msg);
    msg_len = abcdk_comm_message_size(msg);

    ABCDK_PTR2U32(msg_ptr, 0) = abcdk_endian_h_to_b32(msg_len);
    ABCDK_PTR2U32(msg_ptr, 4) = abcdk_endian_h_to_b32(ABCDK_COMM_EASY_MD_PROTOCOL);
    ABCDK_PTR2U64(msg_ptr, 8) = abcdk_endian_h_to_b64(num);
    ABCDK_PTR2U8(msg_ptr, 17) = flag;
    memcpy(ABCDK_PTR2VPTR(msg_ptr, ABCDK_COMM_EASY_MD_HDR_SIZE), cargo, len);

    chk = abcdk_comm_queue_push(easy_p->out_queue, msg);
    if (chk != 0)
        goto final_error;

    if (abcdk_atomic_load(&easy_p->status) == 2)
        abcdk_comm_write_watch(node);

    return 0;

final_error:

    abcdk_comm_message_unref(&msg);

    return -1;
}

abcdk_comm_message_t *_abcdk_comm_easy_extrac_cargo(abcdk_comm_message_t *msg)
{
    abcdk_comm_message_t *cargo;
    void *msg_ptr;
    size_t msg_len;
    void *cargo_ptr;
    size_t cargo_len;
    int chk;

    msg_ptr = abcdk_comm_message_data(msg);
    msg_len = abcdk_comm_message_size(msg);

    cargo = abcdk_comm_message_alloc(msg_len - ABCDK_COMM_EASY_MD_HDR_SIZE);
    if(!cargo)
        return NULL;

    cargo_ptr = abcdk_comm_message_data(cargo);
    cargo_len = abcdk_comm_message_size(cargo);

    memcpy(cargo_ptr, ABCDK_PTR2VPTR(msg_ptr, ABCDK_COMM_EASY_MD_HDR_SIZE), cargo_len);

    return cargo;
}

int abcdk_comm_easy_state(abcdk_comm_node_t *node)
{
    abcdk_comm_easy_t *easy_p = NULL;

    assert(node != NULL);

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(easy_p->magic == ABCDK_COMM_EASY_MAGIC,"未通过easy接口建立连接，不能调此接口。");
    

    if (!abcdk_atomic_load(&easy_p->status))
        return -1;

    return 0;
}


int abcdk_comm_easy_request(abcdk_comm_node_t *node, const void *data, size_t len, abcdk_comm_message_t **rsp)
{
    abcdk_comm_easy_t *easy_p = NULL;
    abcdk_comm_queue_t *rsp_queue = NULL;
    abcdk_comm_message_t *rsp_msg = NULL;
    uint64_t mid;
    int chk;

    assert(node != NULL && data != NULL && len > 0);
    assert(len <= ABCDK_COMM_EASY_MD_MAX_SIZE - ABCDK_COMM_EASY_MD_HDR_SIZE);

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(easy_p->magic == ABCDK_COMM_EASY_MAGIC,"未通过easy接口建立连接，不能调此接口。");

    if(!abcdk_atomic_load(&easy_p->status))
        return -2;

    mid = _abcdk_comm_easy_make_mid();

    if (rsp)
        abcdk_comm_waiter_request2(easy_p->rsp_waiter, &mid);

    /*发送请求(仅向输出队列注册事件和消息)。*/
    chk = _abcdk_comm_easy_post(node, data, len, mid, 0);
    if (chk != 0)
        return -1;

    /*不需要应答直接返回。*/
    if (!rsp)
        return 0;

    rsp_queue = abcdk_comm_waiter_wait2(easy_p->rsp_waiter, &mid, 1, INTMAX_MAX);
    if (!rsp_queue)
        return -1;

    rsp_msg = abcdk_comm_queue_pop(rsp_queue);
    abcdk_comm_queue_free(&rsp_queue);

    if (!rsp_msg)
        return -2;

    /*提取应答货物(应用层数据包)。*/
    *rsp = _abcdk_comm_easy_extrac_cargo(rsp_msg);
    abcdk_comm_message_unref(&rsp_msg);

    return 0;
}

int abcdk_comm_easy_response(abcdk_comm_node_t *node, const void *data, size_t len)
{
    abcdk_comm_easy_t *easy_p = NULL;
    uint64_t *mid_p;
    int chk;

    assert(node != NULL && data != NULL && len > 0);
    assert(len <= ABCDK_COMM_EASY_MD_MAX_SIZE - ABCDK_COMM_EASY_MD_HDR_SIZE);

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);
    ABCDK_ASSERT(easy_p->magic == ABCDK_COMM_EASY_MAGIC,"未通过easy接口建立连接，不能调此接口。");

    if(!abcdk_atomic_load(&easy_p->status))
        return -2;

    /*获取请求MID。*/
    mid_p = pthread_getspecific(easy_p->req_ptkey);
    ABCDK_ASSERT(mid_p != NULL,"每次请求仅允许应答一次。");

    chk = _abcdk_comm_easy_post(node, data, len, *mid_p, ABCDK_COMM_EASY_MD_FLAG_RSP);
    if (chk != 0)
        return -1;  

    /*解除线程的MID(应答一次就好)。*/
    pthread_setspecific(easy_p->req_ptkey, NULL); 

    return 0;
}

void _abcdk_comm_easy_event_accept(abcdk_comm_node_t *node, abcdk_comm_node_t *listen)
{
    abcdk_comm_easy_t *listen_easy_p = NULL;
    abcdk_object_t *append_p = NULL;
    abcdk_comm_easy_t *easy_p = NULL;

    listen_easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(listen);

    easy_p = _abcdk_comm_easy_alloc();
    if (!easy_p)
        return;

    append_p = abcdk_comm_node_append(node);
    append_p->pptrs[0] = (uint8_t*)easy_p;
    abcdk_object_atfree(append_p,_abcdk_comm_easy_destroy_cb,NULL);
    abcdk_object_unref(&append_p);

    easy_p->flag = 2;

    /*复制请求回调函数指针。*/
    easy_p->request_cb = listen_easy_p->request_cb;

    /*复制监听的用户环境指针。*/
    abcdk_comm_set_userdata(node,abcdk_comm_get_userdata(listen));

}

void _abcdk_comm_easy_event_connect(abcdk_comm_node_t *node)
{
    abcdk_comm_easy_t *easy_p = NULL;

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);

    abcdk_atomic_store(&easy_p->status, 2);

    /*已连接到远端，注册读写事件。*/
    abcdk_comm_read_watch(node);
    abcdk_comm_write_watch(node);
}

int _abcdk_comm_easy_msg_protocol(abcdk_comm_node_t *node, abcdk_comm_message_t *msg)
{
    uint32_t len;
    uint32_t pro;
    size_t off;
    
    off = abcdk_comm_message_offset(msg);
    if (off < ABCDK_COMM_EASY_MD_HDR_SIZE)
        return 0;

    len = abcdk_endian_b_to_h32(ABCDK_PTR2U32(abcdk_comm_message_data(msg), 0));
    pro = abcdk_endian_b_to_h32(ABCDK_PTR2U32(abcdk_comm_message_data(msg), 4));

    /*不支持太大的数据包。*/
    if (len > ABCDK_COMM_EASY_MD_MAX_SIZE)
        return -1;

    /*仅支持相同的协议。*/
    if(pro != ABCDK_COMM_EASY_MD_PROTOCOL)
        return -1;

    if (len != abcdk_comm_message_size(msg))
    {
        abcdk_comm_message_realloc(msg, len);
        return 0;
    }
    else if (len != abcdk_comm_message_offset(msg))
    {
        return 0;
    }

    return 1;
}

void _abcdk_comm_easy_event_input(abcdk_comm_node_t *node)
{
    abcdk_comm_easy_t *easy_p = NULL;
    abcdk_comm_message_t *msg = NULL;
    void *msg_ptr;
    size_t msg_len;
    uint64_t mid;
    uint8_t flag;
    void *cargo_ptr;
    size_t cargo_len;
    int chk;

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);

    /*准备接收数的缓存。*/
    if (!easy_p->in_buffer)
    {
        easy_p->in_buffer = abcdk_comm_message_alloc(ABCDK_COMM_EASY_MD_HDR_SIZE);
        abcdk_comm_message_protocol_set(easy_p->in_buffer, _abcdk_comm_easy_msg_protocol);
    }

    /*没有可用的缓存时，通知超时，以关闭这个连接。*/
    if (!easy_p->in_buffer)
    {
        abcdk_comm_set_timeout(node, 1);
        return;
    }

    chk = abcdk_comm_message_recv(node, easy_p->in_buffer);
    if (chk < 0)
    {
        abcdk_comm_set_timeout(node, 1);
        return;
    }
    else if (chk == 0)
    {
        abcdk_comm_read_watch(node);
        return;
    }

    /*托管缓存。*/
    msg = easy_p->in_buffer;
    /*缓存已经被托管，这里不能再继续使用了。*/
    easy_p->in_buffer = NULL;
    
    /*复用链路前要增加引用计数，以防止多线程操作同一个链路在释放回收内存后，造成应用层内存非法访问的异常。*/
    abcdk_comm_node_refer(node);
    /*复用链路。*/
    abcdk_comm_read_watch(node);

    /*处理接收到的数据。*/
    msg_ptr = abcdk_comm_message_data(msg);
    msg_len = abcdk_comm_message_size(msg);

    mid = abcdk_endian_b_to_h64(ABCDK_PTR2U64(msg_ptr, 8));
    flag = ABCDK_PTR2U8(msg_ptr, 17);

    cargo_ptr = ABCDK_PTR2VPTR(msg_ptr, ABCDK_COMM_EASY_MD_HDR_SIZE);
    cargo_len = msg_len - ABCDK_COMM_EASY_MD_HDR_SIZE;

    /*检测是请求还是应答。*/
    if (flag & ABCDK_COMM_EASY_MD_FLAG_RSP)
    {
        abcdk_comm_waiter_response2(easy_p->rsp_waiter, &mid, msg);
    }
    else
    {
        /*绑定线程的MID(用于应答)。*/
        pthread_setspecific(easy_p->req_ptkey, &mid);

        /*通知应用层，数据到达。*/
        if (easy_p->request_cb)
            easy_p->request_cb(node,cargo_ptr,cargo_len);

        /*解除线程的MID。*/
        pthread_setspecific(easy_p->req_ptkey, NULL);

        /*删除请求数据。*/
        abcdk_comm_message_unref(&msg);
    }

    /*减少引用计数。*/
    abcdk_comm_node_unref(&node);
}

void _abcdk_comm_easy_event_output(abcdk_comm_node_t *node)
{
    abcdk_comm_easy_t *easy_p = NULL;
    int chk;

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);

NEXT_MSG:

    if (!easy_p->out_buffer)
    {
        easy_p->out_buffer = abcdk_comm_queue_pop(easy_p->out_queue);
        if (!easy_p->out_buffer)
            return;
    }

    chk = abcdk_comm_message_send(node, easy_p->out_buffer);
    if (chk < 0)
    {
        abcdk_comm_set_timeout(node, 1);
        return;
    }
    else if (chk == 0)
    {
        abcdk_comm_write_watch(node);
        return;
    }

    /*释放消息缓存，并继续发送。*/
    abcdk_comm_message_unref(&easy_p->out_buffer);
    goto NEXT_MSG;
}

void _abcdk_comm_easy_event_close(abcdk_comm_node_t *node)
{
    abcdk_comm_easy_t *easy_p = NULL;
    char sockname_str[NAME_MAX] = {0};
    char peername_str[NAME_MAX] = {0};

    easy_p = (abcdk_comm_easy_t *)abcdk_comm_get_append(node);

    if (easy_p)
    {
        abcdk_atomic_store(&easy_p->status, 0);

        /*通知所有在这个线路上等待应答的请求，连接已经关闭。*/
        abcdk_comm_waiter_cancel(easy_p->rsp_waiter);

        /*通知连接已断开。*/
        if (easy_p->request_cb)
            easy_p->request_cb(node, NULL, 0);

        // /*断开后，做最后的清理工作。*/
        // abcdk_comm_node_unref(&node);
    }
    else
    {
        /*ACCEPT可能还未完成连接就已经断开了。*/

        abcdk_comm_get_sockaddr_str(node, sockname_str, peername_str);
        fprintf(stderr, "sockname(%s) -> peername(%s) disconnected.\n",sockname_str, peername_str);
    }
}

void _abcdk_comm_easy_event_cb(abcdk_comm_node_t *node, uint32_t event, abcdk_comm_node_t *listen)
{
    switch (event)
    {
    case ABCDK_COMM_EVENT_ACCEPT:
        _abcdk_comm_easy_event_accept(node,listen);
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
    case ABCDK_COMM_EVENT_LISTEN_CLOSE:
    default:
        _abcdk_comm_easy_event_close(node);
        break;
    }
}


int abcdk_comm_easy_listen(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_comm_easy_request_cb request_cb)
{
    abcdk_comm_easy_t *easy = NULL;
    abcdk_object_t *append_p = NULL;
    int chk;

    assert(node != NULL && addr != NULL && request_cb != NULL);

    easy = _abcdk_comm_easy_alloc();
    if (!easy)
        goto final_error;
    
    append_p = abcdk_comm_node_append(node);
    append_p->pptrs[0] = (uint8_t*)easy;
    abcdk_object_atfree(append_p,_abcdk_comm_easy_destroy_cb,NULL);
    abcdk_object_unref(&append_p);

    easy->flag = 3;
    easy->status = 2;
    easy->request_cb = request_cb;

    chk = abcdk_comm_listen(node, ssl_ctx, addr, _abcdk_comm_easy_event_cb);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    return -1;
}

int abcdk_comm_easy_connect(abcdk_comm_node_t *node, SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr, abcdk_comm_easy_request_cb request_cb)
{
    abcdk_comm_easy_t *easy = NULL;
    abcdk_object_t *append_p = NULL;
    int chk;

    assert(node != NULL && addr != NULL && request_cb != NULL);

    easy = _abcdk_comm_easy_alloc();
    if (!easy)
        goto final_error;
    
    append_p = abcdk_comm_node_append(node);
    append_p->pptrs[0] = (uint8_t*)easy;
    abcdk_object_atfree(append_p,_abcdk_comm_easy_destroy_cb,NULL);
    abcdk_object_unref(&append_p);

    easy->flag = 1;
    easy->status = 1;
    easy->request_cb = request_cb;

    chk = abcdk_comm_connect(node, ssl_ctx, addr, _abcdk_comm_easy_event_cb);
    if (chk != 0)
        goto final_error;

    return 0;

final_error:

    return -1;
}