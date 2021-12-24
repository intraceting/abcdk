/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk-comm/easy.h"

/*
 * --------------------------------------------------------
 * |Message Data                                          |
 * --------------------------------------------------------
 * |Length  |Protocol |Number  |Flag    |Reserve |Cargo   |
 * |4 Bytes |4 Bytes  |8 Bytes |1 Bytes |3 Bytes |N Bytes |
 * --------------------------------------------------------
*/

/** 数据包头部长度。*/
#define ABCDK_COMM_EASY_MD_HDR_SIZE (20)
/** 数据包最大长度。*/
#define ABCDK_COMM_EASY_MD_MAX_SIZE ((256*1024*1024)-1)

/** 应答标志。*/
#define ABCDK_COMM_EASY_MD_FLAG_RSP 0x01

/** 简单通信节点。*/
typedef struct _abcdk_comm_easy 
{
    /** 引用计数器。*/
    volatile int refcount;

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

    /** 本机地址。*/
    abcdk_sockaddr_t local;

    /** 远端地址。*/
    abcdk_sockaddr_t remote;

    /** 链路。*/
    abcdk_comm_node_t *comm;

    /** 请求回调函数指针。*/
    abcdk_comm_easy_request_cb request_cb;

    /** 应用层环境指针。*/
    void *opaque;

    /** 输入消息缓存。*/
    abcdk_comm_message_t *in_buffer;

    /** 输出消息缓存。*/
    abcdk_comm_message_t *out_buffer;

    /** 输出消息队列。*/
    abcdk_comm_queue_t *out_queue;

    /** 应答服务员。*/
    abcdk_comm_waiter_t *rsp_waiter;

}abcdk_comm_easy_t;


void abcdk_comm_easy_unref(abcdk_comm_easy_t **easy)
{
    abcdk_comm_easy_t *easy_p = NULL;

    if (!easy || !*easy)
        return;

    easy_p = *easy;

    if (abcdk_atomic_fetch_and_add(&easy_p->refcount, -1) != 1)
        goto final;

    assert(easy_p->refcount == 0);

    abcdk_comm_node_unref(&easy_p->comm);
    abcdk_comm_message_unref(&easy_p->in_buffer);
    abcdk_comm_queue_free(&easy_p->out_queue);
    abcdk_comm_waiter_free(&easy_p->rsp_waiter);

    abcdk_heap_free(easy_p);

final:

    /*set NULL(0).*/
    *easy = NULL;
}

abcdk_comm_easy_t *abcdk_comm_easy_refer(abcdk_comm_easy_t *src)
{
    int chk;

    assert(src != NULL);

    chk = abcdk_atomic_fetch_and_add(&src->refcount, 1);
    assert(chk > 0);

    return src;
}

abcdk_comm_easy_t *_abcdk_comm_easy_alloc()
{
    abcdk_comm_easy_t *easy;

    easy = (abcdk_comm_easy_t*)abcdk_heap_alloc(sizeof(abcdk_comm_easy_t));
    if(!easy)
        return NULL;

    easy->flag = 0;
    easy->status = 1;

    easy->in_buffer = NULL;
    easy->out_buffer = NULL;
    easy->out_queue = abcdk_comm_queue_alloc();
    easy->rsp_waiter = abcdk_comm_waiter_alloc();

    return easy;
}

int abcdk_comm_easy_set_timeout(abcdk_comm_easy_t *easy, time_t timeout)
{
    int chk;

    assert(easy != NULL);

    if (abcdk_atomic_load(&easy->status) != 2)
        return -1;

    chk = abcdk_comm_set_timeout(easy->comm,timeout);
    if(chk == 0)
        return 0;

    return -1;
}

int abcdk_comm_easy_get_sockname(abcdk_comm_easy_t *easy, abcdk_sockaddr_t *addr)
{
    assert(easy != NULL);

    *addr = easy->local;

    return 0;
}

int abcdk_comm_easy_get_peername(abcdk_comm_easy_t *easy, abcdk_sockaddr_t *addr)
{
    assert(easy != NULL);

    *addr = easy->remote;

    return 0;
}

void *abcdk_comm_easy_set_userdata(abcdk_comm_easy_t *easy, void *opaque)
{
    void *old = NULL;

    assert(easy != NULL);

    old = easy->opaque;
    easy->opaque = opaque;
    
    return old;
}

void *abcdk_comm_easy_get_userdata(abcdk_comm_easy_t *easy)
{
    void *old = NULL;

    assert(easy != NULL);

    old = easy->opaque;
    
    return old;
}

uint64_t _abcdk_comm_easy_make_mid()
{
    static volatile uint64_t mid = 1;

    return abcdk_atomic_fetch_and_add(&mid,1);
}

int _abcdk_comm_easy_post(abcdk_comm_easy_t *easy, abcdk_comm_message_t *cargo,uint64_t num,uint8_t flag)
{
    abcdk_comm_message_t *msg = NULL;
    void *msg_ptr;
    size_t msg_len;
    void *cargo_ptr;
    size_t cargo_len;
    int chk;

    cargo_ptr = abcdk_comm_message_data(cargo);
    cargo_len = abcdk_comm_message_size(cargo);

    msg = abcdk_comm_message_alloc(4 + 4 + 8 + 4 + cargo_len);
    if (!msg)
        goto final_error;

    msg_ptr = abcdk_comm_message_data(msg);
    msg_len = abcdk_comm_message_size(msg);

    ABCDK_PTR2U32(msg_ptr,0) = abcdk_endian_h_to_b32(msg_len);
    ABCDK_PTR2U32(msg_ptr,4) = abcdk_endian_h_to_b32(123456789);
    ABCDK_PTR2U64(msg_ptr,8) = abcdk_endian_h_to_b64(num);
    ABCDK_PTR2U8(msg_ptr,17) = flag;
    memcpy(ABCDK_PTR2VPTR(msg_ptr, ABCDK_COMM_EASY_MD_HDR_SIZE), cargo_ptr, cargo_len);

    chk = abcdk_comm_queue_push(easy->out_queue,msg);
    if(chk != 0)
        goto final_error;

    if(abcdk_atomic_load(&easy->status) == 2)
        abcdk_comm_write_watch(easy->comm);

    return 0;

final_error:

    abcdk_comm_message_unref(&msg);

    return -1;
}

int _abcdk_comm_easy_extrac_cargo(abcdk_comm_message_t *msg, abcdk_comm_message_t *cargo)
{
    void *msg_ptr;
    size_t msg_len;
    void *cargo_ptr;
    size_t cargo_len;
    int chk;

    msg_ptr = abcdk_comm_message_data(msg);
    msg_len = abcdk_comm_message_size(msg);

    chk = abcdk_comm_message_realloc(cargo, msg_len - ABCDK_COMM_EASY_MD_HDR_SIZE);
    if (chk != 0)
        return -1;
    
    cargo_ptr = abcdk_comm_message_data(cargo);
    cargo_len = abcdk_comm_message_size(cargo);

    memcpy(cargo_ptr,ABCDK_PTR2VPTR(msg_ptr, ABCDK_COMM_EASY_MD_HDR_SIZE), cargo_len);

    return 0;
}

int abcdk_comm_easy_request(abcdk_comm_easy_t *easy, abcdk_comm_message_t *req,
                            abcdk_comm_message_t **rsp, time_t timeout)
{
    abcdk_comm_queue_t *rsp_queue = NULL;
    abcdk_comm_message_t *rsp_msg = NULL;
    uint64_t mid;
    int chk;

    assert(easy != NULL && req != NULL);

    mid = _abcdk_comm_easy_make_mid();

    if(rsp)
        abcdk_comm_waiter_request2(easy->rsp_waiter,&mid);

    chk = _abcdk_comm_easy_post(easy, req, mid, 0);
    if (chk != 0)
        return -1;

    if(rsp)
    {
        rsp_queue = abcdk_comm_waiter_wait2(easy->rsp_waiter,&mid,1,timeout);
        if(!rsp_queue)
            return -1;
    }

    rsp_msg = abcdk_comm_queue_pop(rsp_queue);
    if(rsp_msg)
    {
        _abcdk_comm_easy_extrac_cargo(rsp_msg,*rsp);
        abcdk_comm_message_unref(&rsp_msg);
    }

    abcdk_comm_queue_free(&rsp_queue);

    return 0;
}

void _abcdk_comm_easy_event_accept(abcdk_comm_node_t *node)
{
    abcdk_comm_easy_t *easy_listen = (abcdk_comm_easy_t*)abcdk_comm_get_userdata(node);
    abcdk_comm_easy_t *easy;

    easy = _abcdk_comm_easy_alloc();
    if(!easy)
    {
        abcdk_comm_set_timeout(node,1);
        return;
    }

    easy->flag = 2;
    easy->comm = abcdk_comm_node_refer(node);
    abcdk_atomic_store(&easy->status,2);
    easy->request_cb = easy_listen->request_cb;
    easy->opaque = easy_listen->opaque;
    
    abcdk_comm_get_sockname(node, &easy->local);
    abcdk_comm_get_peername(node, &easy->remote);

    /*替换环境指针*/
    abcdk_comm_set_userdata(node,easy);

    abcdk_comm_read_watch(node);
}

void _abcdk_comm_easy_event_connect(abcdk_comm_node_t *node)
{
    abcdk_comm_easy_t *easy = (abcdk_comm_easy_t*)abcdk_comm_get_userdata(node);

    easy->comm = abcdk_comm_node_refer(node);
    abcdk_atomic_store(&easy->status,2);
    
    abcdk_comm_get_sockname(node, &easy->local);
    abcdk_comm_get_peername(node, &easy->remote);

    abcdk_comm_read_watch(node);
}

int _abcdk_comm_easy_msg_protocol(abcdk_comm_node_t *node, abcdk_comm_message_t *msg)
{
    size_t off = abcdk_comm_message_offset(msg);
    if (off < 4)
        return 0;

    size_t len = abcdk_endian_b_to_h32(ABCDK_PTR2U32(abcdk_comm_message_data(msg), 0));
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
    abcdk_comm_easy_t *easy = (abcdk_comm_easy_t*)abcdk_comm_get_userdata(node);
    void *msg_ptr;
    size_t msg_len;
    uint64_t mid;
    uint8_t flag;
    abcdk_comm_message_t *msg;
    abcdk_comm_message_t *req_cargo;
    abcdk_comm_message_t *rsp_cargo;
    int chk;
    
    /*准备接收数的缓存。*/
    if(!easy->in_buffer)
    {
        easy->in_buffer = abcdk_comm_message_alloc(4);
        abcdk_comm_message_protocol_set(easy->in_buffer,_abcdk_comm_easy_msg_protocol);
    }

    /*没有可用的缓存时，通知超时，以关闭这个连接。*/
    if(!easy->in_buffer)
    {
        abcdk_comm_set_timeout(easy->comm, 1);
        return;
    }

    chk = abcdk_comm_message_recv(easy->comm, easy->in_buffer);
    if (chk != 1)
    {
        abcdk_comm_read_watch(easy->comm);
        return;
    }
    
        msg_ptr = abcdk_comm_message_data(easy->in_buffer);
        msg_len = abcdk_comm_message_size(easy->in_buffer);

        mid = abcdk_endian_b_to_h64(ABCDK_PTR2U64(msg_ptr,8));
        flag = ABCDK_PTR2U8(msg_ptr,17);

        /*检测是请求还是应答。*/
        if (flag & ABCDK_COMM_EASY_MD_FLAG_RSP)
        {
            abcdk_comm_waiter_response2(easy->rsp_waiter, &mid, easy->in_buffer);

            /*缓存已经被托管，这里不能再继续使用了。*/
            easy->in_buffer = NULL;
            abcdk_comm_read_watch(easy->comm);
        }
        else
        {
            /*提取请求数据。*/
            chk = _abcdk_comm_easy_extrac_cargo(easy->in_buffer, req_cargo);
            if (chk != 0)
            {
                abcdk_comm_set_timeout(easy->comm, 1);
            }
            else
            {
                /*复用缓存。*/
                abcdk_comm_message_realloc(easy->in_buffer, 4);
                abcdk_comm_message_reset(easy->in_buffer);
                /*复用链路。*/
                abcdk_comm_read_watch(easy->comm);

                /*通知应用层，数据到达。*/
                if (easy->request_cb)
                    easy->request_cb(easy, req_cargo, &rsp_cargo);

                if (rsp_cargo)
                {
                    chk = _abcdk_comm_easy_post(easy, rsp_cargo, mid, ABCDK_COMM_EASY_MD_FLAG_RSP);
                    if (chk != 0)
                    {
                        abcdk_comm_message_unref(&rsp_cargo);
                    }
                }
            }
        }
    
}


void _abcdk_comm_easy_event_output(abcdk_comm_node_t *node)
{
    abcdk_comm_easy_t *easy = (abcdk_comm_easy_t *)abcdk_comm_get_userdata(node);

NEXT_MSG:

    if (!easy->out_buffer)
    {
        easy->out_buffer = abcdk_comm_queue_pop(easy->out_queue);
        if (!easy->out_buffer)
            return;
    }

    chk = abcdk_comm_message_send(easy->comm, easy->out_buffer);
    if (chk < 0)
    {
        abcdk_comm_set_timeout(easy->comm, 1);
        return;
    }
    else if (chk == 0)
    {
        abcdk_comm_write_watch(easy->comm);
        return;
    }

    /*释放消息缓存，并继续发送。*/
    abcdk_comm_message_unref(&easy->out_buffer);
    goto NEXT_MSG;
}

void _abcdk_comm_easy_event_close(abcdk_comm_node_t *node)
{
    abcdk_comm_easy_t *easy = (abcdk_comm_easy_t *)abcdk_comm_get_userdata(node);
    abcdk_sockaddr_t sockname, peername;
    char sockname_str[100] = {0},peername_str[100] = {0};

    if (easy)
    {
        abcdk_atomic_store(&easy->status, 0);

        /*通知连接已断开。*/
        if(easy->request_cb)
            easy->request_cb(easy, NULL, NULL);

        abcdk_comm_easy_unref(&easy);
    }
    else
    {
        /*可能还未完成连接就已经断开了。*/

        abcdk_comm_get_sockname(node, &sockname);
        abcdk_comm_get_peername(node, &peername);

        if (sockname.family)
            abcdk_sockaddr_to_string(sockname_str, &sockname);
        if (peername.family)
            abcdk_sockaddr_to_string(peername_str, &peername);

        syslog(LOG_WARNING,"Socket: %s -> %s Disconnected.\n", sockname_str, peername_str);
    }
}

void _abcdk_comm_easy_event_cb(abcdk_comm_node_t *node, uint32_t event)
{
    switch (event)
    {
    case ABCDK_COMM_EVENT_ACCEPT:
        _abcdk_comm_easy_event_accept(node);
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

abcdk_comm_easy_t *abcdk_comm_easy_listen(SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr,
                                          abcdk_comm_easy_request_cb request_cb, void *opaque)
{
    abcdk_comm_easy_t *easy = NULL;
    abcdk_comm_easy_t *easy_p = NULL;
    int chk;

    assert(addr != NULL && request_cb != NULL);

    easy = _abcdk_comm_easy_alloc();
    if(!easy)
        return NULL;

    /*应用层需要保持这个对象引用。*/
    easy_p = abcdk_comm_easy_refer(easy);

    easy->flag = 3;
    easy->status = 2;
    easy->local = *addr;
    easy->request_cb = request_cb;
    easy->opaque = opaque;

    chk = abcdk_comm_listen(ssl_ctx,&easy->local,_abcdk_comm_easy_event_cb,easy);
    if(chk != 0)
        goto final_error;

    return easy_p;

final_error:

    abcdk_comm_easy_unref(&easy);
    abcdk_comm_easy_unref(&easy_p);

    return NULL;
}

abcdk_comm_easy_t *abcdk_comm_easy_connect(SSL_CTX *ssl_ctx, abcdk_sockaddr_t *addr,
                                           abcdk_comm_easy_request_cb request_cb, void *opaque)
{
    abcdk_comm_easy_t *easy = NULL;
    abcdk_comm_easy_t *easy_p = NULL;
    int chk;

    assert(addr != NULL && request_cb != NULL);

    easy = _abcdk_comm_easy_alloc();
    if(!easy)
        return NULL;

    /*应用层需要保持这个对象引用。*/
    easy_p = abcdk_comm_easy_refer(easy);

    easy->flag = 1;
    easy->status = 1;
    easy->remote = *addr;
    easy->request_cb = request_cb;
    easy->opaque = opaque;

    chk = abcdk_comm_connect(ssl_ctx,&easy->local,_abcdk_comm_easy_event_cb,easy);
    if(chk != 0)
        goto final_error;

    return easy_p;

final_error:

    abcdk_comm_easy_unref(&easy);
    abcdk_comm_easy_unref(&easy_p);

    return NULL;   
}