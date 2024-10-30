/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/net/srpc.h"

/**简单的RPC服务。*/
struct _abcdk_srpc
{
    /*通讯IO。*/
    abcdk_stcp_t *io_ctx;

    /*请求队列。*/
    abcdk_worker_t *req_list;

};//abcdk_srpc_t

/**RPC会话(内部)。*/
typedef struct _abcdk_srpc_node
{
    /*父级。*/
    abcdk_srpc_t *father;

    /*配置。*/
    abcdk_srpc_config_t cfg;

    /*标志。0 监听，1 服务端，2 客户端。*/
    int flag;

    /*远程地址。*/
    char remote_addr[NAME_MAX];

    /*本机地址。*/
    char local_addr[NAME_MAX];

    /*请求数据。*/
    abcdk_receiver_t *req_data;

    /*请求服务员。*/
    abcdk_waiter_t *req_waiter;

    /*消息编号*/
    uint64_t mid_next;

    /** 用户环境指针。*/
    abcdk_object_t *userdata;

    /** 用户环境销毁函数。*/
    void (*userdata_free_cb)(void *userdata);

} abcdk_srpc_node_t;

/**RPC请求项目。*/
typedef struct _abcdk_srpc_request_item
{
    /*STCP节点。*/
    abcdk_stcp_node_t *node;

    /*请求数据。*/
    abcdk_receiver_t *req_data;
    
}abcdk_srpc_request_item_t;


void abcdk_srpc_unref(abcdk_srpc_session_t **session)
{
    abcdk_stcp_unref((abcdk_stcp_node_t**)session);
}


abcdk_srpc_session_t *abcdk_srpc_refer(abcdk_srpc_session_t *src)
{
    return (abcdk_srpc_session_t*)abcdk_stcp_refer((abcdk_stcp_node_t*)src);
}

static void _abcdk_srpc_node_destroy_cb(void *userdata)
{
    abcdk_srpc_node_t *node_p;

    if (!userdata)
        return;

    node_p = (abcdk_srpc_node_t *)userdata;

    /*通知应用层回收内存。*/
    if(node_p->userdata_free_cb)
        node_p->userdata_free_cb(node_p->userdata->pptrs[0]);

    abcdk_object_unref(&node_p->userdata);
    abcdk_receiver_unref(&node_p->req_data);
    abcdk_waiter_free(&node_p->req_waiter);

}

static void _abcdk_srpc_node_waiter_msg_destroy_cb(void *msg)
{
    abcdk_object_unref((abcdk_object_t**)&msg);
}

abcdk_srpc_session_t *abcdk_srpc_alloc(abcdk_srpc_t *ctx, size_t userdata, void (*free_cb)(void *userdata))
{
    abcdk_stcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;

    assert(ctx != NULL && free_cb != NULL);

    node_p = abcdk_stcp_alloc(ctx->io_ctx, sizeof(abcdk_srpc_node_t), _abcdk_srpc_node_destroy_cb);
    if (!node_p)
        return NULL;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node_p);

    node_ctx_p->father = ctx;
    node_ctx_p->userdata = abcdk_object_alloc3(userdata,1);
    node_ctx_p->userdata_free_cb = free_cb;
    node_ctx_p->req_waiter = abcdk_waiter_alloc(_abcdk_srpc_node_waiter_msg_destroy_cb);
    node_ctx_p->mid_next = 1;

    return (abcdk_srpc_session_t*)node_p;
}

void abcdk_srpc_trace_output(abcdk_srpc_session_t *node,int type, const char* fmt,...)
{
    abcdk_stcp_node_t *node_p;
    char new_tname[18] = {0}, old_tname[18] = {0};

    node_p = (abcdk_stcp_node_t *)node;

    snprintf(new_tname, 16, "%x", abcdk_stcp_get_index(node_p));

#ifdef __USE_GNU
    pthread_getname_np(pthread_self(), old_tname, 18);
    pthread_setname_np(pthread_self(), new_tname);
#endif //__USE_GNU

    va_list vp;
    va_start(vp, fmt);
    abcdk_trace_voutput(type, fmt, vp);
    va_end(vp);

#ifdef __USE_GNU
    pthread_setname_np(pthread_self(), old_tname);
#endif //__USE_GNU
}

uint64_t abcdk_srpc_get_index(abcdk_srpc_session_t *node)
{
    abcdk_stcp_node_t *node_p;

    assert(node != NULL);

    node_p = (abcdk_stcp_node_t *)node;

    return abcdk_stcp_get_index(node_p);
}

void *abcdk_srpc_get_userdata(abcdk_srpc_session_t *session)
{
    abcdk_stcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;

    assert(session != NULL);

    node_p = (abcdk_stcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node_p);

    return node_ctx_p->userdata->pptrs[0];
}

const char *abcdk_srpc_get_address(abcdk_srpc_session_t *session, int remote)
{
    abcdk_stcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    void *old_userdata;

    assert(session != NULL);

    node_p = (abcdk_stcp_node_t *)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node_p);

    if (remote)
        return node_ctx_p->remote_addr;
    else
        return node_ctx_p->local_addr;
}

void abcdk_srpc_set_timeout(abcdk_srpc_session_t *session,time_t timeout)
{
    abcdk_stcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    void *old_userdata;

    assert(session != NULL);

    node_p = (abcdk_stcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node_p);

    abcdk_stcp_set_timeout(node_p,timeout);
}

void abcdk_srpc_destroy(abcdk_srpc_t **ctx)
{
    abcdk_srpc_t *ctx_p;

    if(!ctx ||!*ctx)
        return;

    ctx_p = *ctx;

    abcdk_worker_stop(&ctx_p->req_list);
    abcdk_stcp_stop(&ctx_p->io_ctx);
    abcdk_heap_free(ctx_p);

    /*一定要等STCP对象停下来才能清空指针，否则会因为线程调度问题造成引用空指针。*/
    *ctx = NULL;
}

static void _abcdk_srpc_input_transfer_cb(void *opaque,uint64_t event,void *item);

abcdk_srpc_t *abcdk_srpc_create(int worker)
{
    abcdk_worker_config_t req_list_cfg;
    abcdk_srpc_t *ctx;

    ctx = abcdk_heap_alloc(sizeof(abcdk_srpc_t));
    if(!ctx)
        return NULL;

    worker = ABCDK_CLAMP(worker,1,worker);

    ctx->io_ctx = abcdk_stcp_start(worker);
    if (!ctx->io_ctx)
        goto ERR;

    req_list_cfg.numbers = worker;
    req_list_cfg.opaque = ctx;
    req_list_cfg.process_cb = _abcdk_srpc_input_transfer_cb;

    ctx->req_list = abcdk_worker_start(&req_list_cfg);

    return ctx;
ERR:

    abcdk_srpc_destroy(&ctx);

    return NULL;
}

static void _abcdk_srpc_prepare_cb(abcdk_stcp_node_t **node, abcdk_stcp_node_t *listen)
{
    abcdk_stcp_node_t *node_p;
    abcdk_srpc_node_t *listen_ctx_p, *node_ctx_p;
    abcdk_srpc_config_t *cfg_p;
    int chk;

    listen_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(listen);

    listen_ctx_p->cfg.prepare_cb(listen_ctx_p->cfg.opaque,(abcdk_srpc_session_t **)&node_p,(abcdk_srpc_session_t *)listen);
    if (!node_p)
        return;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node_p);

    node_ctx_p->cfg = listen_ctx_p->cfg;
    node_ctx_p->flag = 1;

    cfg_p = &node_ctx_p->cfg;

    /*准备完毕，返回。*/
    *node = node_p;
}

static void _abcdk_srpc_event_accept(abcdk_stcp_node_t *node, int *result)
{
    abcdk_srpc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node);

    /*默认：允许。*/
    *result = 0;

    if(node_ctx_p->cfg.accept_cb)
        node_ctx_p->cfg.accept_cb(node_ctx_p->cfg.opaque, (abcdk_srpc_session_t*)node, result);
    
    if(*result != 0)
        abcdk_stcp_trace_output(node,LOG_INFO, "禁止远端(%s)连接到本机(%s)。", node_ctx_p->remote_addr, node_ctx_p->local_addr);
}

static void _abcdk_srpc_event_connect(abcdk_stcp_node_t *node)
{
    abcdk_srpc_node_t *node_ctx_p;
    SSL *ssl_p;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node);

    abcdk_stcp_trace_output(node,LOG_INFO, "本机(%s)与远端(%s)的连接已建立。",node_ctx_p->local_addr,node_ctx_p->remote_addr);

    
    /*设置超时。*/
    abcdk_stcp_set_timeout(node, 180);

    if(node_ctx_p->cfg.ready_cb)
        node_ctx_p->cfg.ready_cb(node_ctx_p->cfg.opaque, (abcdk_srpc_session_t*)node);

    /*已连接到远端，注册读写事件。*/
    abcdk_stcp_recv_watch(node);
    abcdk_stcp_send_watch(node);
}

static void _abcdk_srpc_event_output(abcdk_stcp_node_t *node)
{
    abcdk_srpc_node_t *node_ctx_p;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node);

    if(node_ctx_p->cfg.output_cb)
        node_ctx_p->cfg.output_cb(node_ctx_p->cfg.opaque, (abcdk_srpc_session_t*)node);
}

static void _abcdk_srpc_event_close(abcdk_stcp_node_t *node)
{
    abcdk_srpc_node_t *node_ctx_p;
    SSL *ssl_p;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node);

    if (node_ctx_p->flag == 0)
    {
        abcdk_stcp_trace_output(node,LOG_INFO, "监听关闭，忽略。");
        return;
    }
    
    abcdk_stcp_trace_output(node,LOG_INFO, "本机(%s)与远端(%s)的连接已断开。", node_ctx_p->local_addr, node_ctx_p->remote_addr);

    /*如果连接关闭则一定要取消等待的事务，否则可能会造成应用层阻塞。*/
    if(node_ctx_p->flag)
        abcdk_waiter_cancel(node_ctx_p->req_waiter);

    if(node_ctx_p->cfg.close_cb)
        node_ctx_p->cfg.close_cb(node_ctx_p->cfg.opaque,(abcdk_srpc_session_t*)node);
}

static void _abcdk_srpc_event_cb(abcdk_stcp_node_t *node, uint32_t event, int *result)
{
    abcdk_srpc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node);

    if(!node_ctx_p->remote_addr[0])
        abcdk_stcp_get_sockaddr_str(node,NULL,node_ctx_p->remote_addr);
    if(!node_ctx_p->local_addr[0])
        abcdk_stcp_get_sockaddr_str(node,node_ctx_p->local_addr,NULL);

    if (event == ABCDK_STCP_EVENT_ACCEPT)
    {
        _abcdk_srpc_event_accept(node,result);
    }
    else if (event == ABCDK_STCP_EVENT_CONNECT)
    {
        _abcdk_srpc_event_connect(node);
    }
    else if (event == ABCDK_STCP_EVENT_INPUT)
    {
    }
    else if (event == ABCDK_STCP_EVENT_OUTPUT)
    {
        _abcdk_srpc_event_output(node);
    }
    else if (event == ABCDK_STCP_EVENT_CLOSE || event == ABCDK_STCP_EVENT_INTERRUPT)
    {
        _abcdk_srpc_event_close(node);
    }
}

static void _abcdk_srpc_request_process(abcdk_srpc_request_item_t *item)
{
    abcdk_srpc_node_t *node_ctx_p;
    const void *req_data;
    size_t req_size;
    uint32_t len;
    uint8_t cmd;
    uint64_t mid;
    abcdk_object_t *cargo;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(item->node);

    req_data = abcdk_receiver_data(item->req_data, 0);
    req_size = abcdk_receiver_length(item->req_data);

    len = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 0, 32);
    cmd = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 32, 8);
    mid = abcdk_bloom_read_number((uint8_t *)req_data, req_size, 40, 64);

    if (cmd == 1) //RSP
    {
        cargo = abcdk_object_copyfrom(ABCDK_PTR2VPTR(req_data, 13), req_size - 13);
        if(!cargo)
            return;

        chk = abcdk_waiter_response(node_ctx_p->req_waiter, mid, cargo);
        if(chk != 0)
            abcdk_object_unref(&cargo);
    }
    else if (cmd == 2) //REQ
    {
        if(node_ctx_p->cfg.request_cb)
            node_ctx_p->cfg.request_cb(node_ctx_p->cfg.opaque,(abcdk_srpc_session_t*)item->node,mid,ABCDK_PTR2VPTR(req_data, 13), req_size - 13);
    }
}

static void _abcdk_srpc_request_item_free(abcdk_srpc_request_item_t **item)
{
    abcdk_srpc_request_item_t *item_p;

    if(!item ||!*item)
        return;

    item_p = *item;
    *item = NULL;

    abcdk_stcp_unref(&item_p->node);
    abcdk_receiver_unref(&item_p->req_data);
    abcdk_heap_free(item_p);
}

static abcdk_srpc_request_item_t *_abcdk_srpc_request_item_alloc()
{
    return (abcdk_srpc_request_item_t *)abcdk_heap_alloc(sizeof(abcdk_srpc_request_item_t));
}

void _abcdk_srpc_input_transfer_cb(void *opaque,uint64_t event,void *item)
{
    abcdk_srpc_t *ctx = (abcdk_srpc_t*)opaque;
    abcdk_srpc_request_item_t *item_p = (abcdk_srpc_request_item_t *)item;

    _abcdk_srpc_request_process(item_p);

    _abcdk_srpc_request_item_free(&item_p);
}

static void _abcdk_srpc_input_dispatch(abcdk_stcp_node_t *node)
{
    abcdk_srpc_node_t *node_ctx_p;
    abcdk_srpc_request_item_t *item;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node);

    item = _abcdk_srpc_request_item_alloc();
    if(!item)
        return;

    item->node = abcdk_stcp_refer(node);
    item->req_data = abcdk_receiver_refer(node_ctx_p->req_data);

    chk = abcdk_worker_dispatch(node_ctx_p->father->req_list,1,item);
    if(chk == 0)
        return;

    _abcdk_srpc_request_item_free(&item);
}

static void _abcdk_srpc_input_cb(abcdk_stcp_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_srpc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node);

    /*默认没有剩余数据。*/
    *remain = 0;

    if (!node_ctx_p->req_data)
        node_ctx_p->req_data = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_SMB, 16*1024*1024, NULL);

    if (!node_ctx_p->req_data)
        goto ERR;

    /*解包。*/
    chk = abcdk_receiver_append(node_ctx_p->req_data, data, size, remain);
    if (chk < 0)
    {
        *remain = 0;
        goto ERR;
    }
    else if (chk == 0) /*数据包不完整，继续接收。*/
        return;

    _abcdk_srpc_input_dispatch(node);

    /*一定要回收。*/
    abcdk_receiver_unref(&node_ctx_p->req_data);

    /*No Error.*/
    return;

ERR:

    abcdk_stcp_set_timeout(node, -1);
}

int abcdk_srpc_listen(abcdk_srpc_session_t *session,abcdk_sockaddr_t *addr,abcdk_srpc_config_t *cfg)
{
    abcdk_stcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    abcdk_stcp_config_t asio_cfg = {0};
    int chk;

    assert(session != NULL && addr != NULL && cfg != NULL);
    assert(cfg->prepare_cb != NULL && cfg->request_cb != NULL);

    node_p = (abcdk_stcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node_p);

    node_ctx_p->cfg = *cfg;
    node_ctx_p->flag = 0;

    asio_cfg.ssl_scheme = cfg->ssl_scheme;
    asio_cfg.pki_ca_file = cfg->pki_ca_file;
    asio_cfg.pki_ca_path = cfg->pki_ca_path;
    asio_cfg.pki_cert_file = cfg->pki_cert_file;
    asio_cfg.pki_key_file = cfg->pki_key_file;
    asio_cfg.pki_check_cert = cfg->pki_check_cert;
    asio_cfg.sk_key_file = cfg->sk_key_file;
    asio_cfg.sk_key_cipher = cfg->sk_key_cipher;

    asio_cfg.prepare_cb = _abcdk_srpc_prepare_cb;
    asio_cfg.event_cb = _abcdk_srpc_event_cb;
    asio_cfg.input_cb = _abcdk_srpc_input_cb;

    chk = abcdk_stcp_listen(node_p,addr,&asio_cfg);
    if(chk != 0)
        return -1;

    return 0;
}

int abcdk_srpc_connect(abcdk_srpc_session_t *session,abcdk_sockaddr_t *addr,abcdk_srpc_config_t *cfg)
{
    abcdk_stcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    abcdk_stcp_config_t asio_cfg = {0};
    int chk;

    assert(session != NULL && addr != NULL && cfg != NULL);
    assert(cfg->prepare_cb != NULL && cfg->request_cb != NULL);

    node_p = (abcdk_stcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node_p);

    node_ctx_p->cfg = *cfg;
    node_ctx_p->flag = 2;

    asio_cfg.ssl_scheme = cfg->ssl_scheme;
    asio_cfg.pki_ca_file = cfg->pki_ca_file;
    asio_cfg.pki_ca_path = cfg->pki_ca_path;
    asio_cfg.pki_cert_file = cfg->pki_cert_file;
    asio_cfg.pki_key_file = cfg->pki_key_file;
    asio_cfg.pki_check_cert = cfg->pki_check_cert;
    asio_cfg.sk_key_file = cfg->sk_key_file;
    asio_cfg.sk_key_cipher = cfg->sk_key_cipher;

    asio_cfg.prepare_cb = _abcdk_srpc_prepare_cb;
    asio_cfg.event_cb = _abcdk_srpc_event_cb;
    asio_cfg.input_cb = _abcdk_srpc_input_cb;

    chk = abcdk_stcp_connect(node_p,addr,&asio_cfg);
    if(chk != 0)
        return -1;

    return 0;
}

static int _abcdk_srpc_post(abcdk_stcp_node_t *node,  uint8_t cmd, uint64_t mid, const void *data, size_t size,int key)
{
    abcdk_srpc_node_t *node_ctx_p;
    abcdk_object_t *msg;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node);

    /*
     * |Length  |CMD    |MID     |Data    |
     * |4 Bytes |1 Byte |8 Bytes |N Bytes |
     *
     * Length：消息长度。注：不包含自身。
     * CMD：1 应答，2 请求。
     * MID：消息ID。
     */

    msg = abcdk_object_alloc2(4 + 1 + 8 + size);
    if (!msg)
        return -1;

    abcdk_bloom_write_number(msg->pptrs[0], msg->sizes[0], 0, 32, msg->sizes[0] - 4);
    abcdk_bloom_write_number(msg->pptrs[0], msg->sizes[0], 32, 8, cmd);
    abcdk_bloom_write_number(msg->pptrs[0], msg->sizes[0], 40, 64, mid);
    memcpy(msg->pptrs[0] + 13, data, size);

    chk = abcdk_stcp_post(node,msg,key);
    if(chk == 0)
        return 0;

    /*如果发送失败则删除消息，避免发生内存泄漏。*/
    abcdk_object_unref(&msg);
    return -2;
}

int abcdk_srpc_request(abcdk_srpc_session_t *session, const void *req, size_t req_size, abcdk_object_t **rsp)
{
    abcdk_stcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    uint64_t mid;
    int chk;

    assert(session != NULL && req != NULL && req_size > 0 && req_size <= 16777215);

    node_p = (abcdk_stcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node_p);

    mid = abcdk_atomic_fetch_and_add(&node_ctx_p->mid_next, 1);

    if (rsp)
    {
        chk = abcdk_waiter_register(node_ctx_p->req_waiter, mid);
        if (chk != 0)
            return -1;
    }

    chk = _abcdk_srpc_post(node_p,2,mid,req,req_size,rsp?1:0);
    if (chk != 0)
        return -2;

    /*如果需要等待应签，这里就可以返回了。*/
    if(!rsp)
        return 0;

    *rsp = (abcdk_object_t *)abcdk_waiter_wait(node_ctx_p->req_waiter, mid, 24 * 60 * 60* 1000L);
    if (!*rsp)
        return -3;

    return 0;
}


int abcdk_srpc_response(abcdk_srpc_session_t *session, uint64_t mid,const void *data,size_t size)
{
    abcdk_stcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    int chk;

    assert(session != NULL && data != NULL && size >0);

    node_p = (abcdk_stcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_stcp_get_userdata(node_p);

    chk = _abcdk_srpc_post(node_p,1,mid,data,size,1);
    if(chk != 0)
        return -1;

    return 0;
}

void abcdk_srpc_output_ready(abcdk_srpc_session_t *session)
{
    abcdk_stcp_node_t *node_p;

    assert(session != NULL);

    node_p = (abcdk_stcp_node_t*)session;

    abcdk_stcp_send_watch(node_p);
}