/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/asio/srpc.h"

/**简单的SRPC服务。*/
struct _abcdk_srpc
{
    /*通讯IO。*/
    abcdk_asynctcp_t *io_ctx;
};//abcdk_srpc_t


typedef struct _abcdk_srpc_node
{
    /*父级。*/
    abcdk_srpc_t *father;

    /*配置。*/
    abcdk_srpc_config_t cfg;

    /*标志。0 监听，1 服务端，2 客户端。*/
    int flag;

    /*OPENSSL环境。*/
    SSL_CTX *openssl_ctx;

    /*EASYSSL环境。*/
    abcdk_easyssl_t *easyssl_ctx;

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

    /*用户环境指针。*/
    void *userdata;

} abcdk_srpc_node_t;


void abcdk_srpc_unref(abcdk_srpc_session_t **session)
{
    abcdk_asynctcp_unref((abcdk_asynctcp_node_t**)session);
}


abcdk_srpc_session_t *abcdk_srpc_refer(abcdk_srpc_session_t *src)
{
    return (abcdk_srpc_session_t*)abcdk_asynctcp_refer((abcdk_asynctcp_node_t*)src);
}

static void _abcdk_srpc_node_destroy_cb(void *userdata)
{
    abcdk_srpc_node_t *ctx;

    if (!userdata)
        return;

    ctx = (abcdk_srpc_node_t *)userdata;

#ifdef HEADER_SSL_H
    abcdk_openssl_ssl_ctx_free(&ctx->openssl_ctx);
#endif //HEADER_SSL_H

    abcdk_easyssl_destroy(&ctx->easyssl_ctx);

    abcdk_receiver_unref(&ctx->req_data);
    abcdk_waiter_free(&ctx->req_waiter);

}

static void _abcdk_srpc_node_waiter_msg_destroy_cb(void *msg)
{
    abcdk_object_unref((abcdk_object_t**)&msg);
}

abcdk_srpc_session_t *abcdk_srpc_alloc(abcdk_srpc_t *ctx)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;

    assert(ctx != NULL);

    node_p = abcdk_asynctcp_alloc(ctx->io_ctx, sizeof(abcdk_srpc_node_t), _abcdk_srpc_node_destroy_cb);
    if (!node_p)
        return NULL;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    node_ctx_p->father = ctx;
    node_ctx_p->req_waiter = abcdk_waiter_alloc(_abcdk_srpc_node_waiter_msg_destroy_cb);

    return (abcdk_srpc_session_t*)node_p;
}

void *abcdk_srpc_get_userdata(abcdk_srpc_session_t *session)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;

    assert(session != NULL);

    node_p = (abcdk_asynctcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    return node_ctx_p->userdata;
}

void *abcdk_srpc_set_userdata(abcdk_srpc_session_t *session,void *userdata)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    void *old_userdata;

    assert(session != NULL);

    node_p = (abcdk_asynctcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    old_userdata = node_ctx_p->userdata;
    node_ctx_p->userdata = userdata;
    
    return old_userdata;
}

const char *abcdk_srpc_get_address(abcdk_srpc_session_t *session, int remote)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    void *old_userdata;

    assert(session != NULL);

    node_p = (abcdk_asynctcp_node_t *)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    if (remote)
        return node_ctx_p->remote_addr;
    else
        return node_ctx_p->local_addr;
}

void abcdk_srpc_set_timeout(abcdk_srpc_session_t *session,time_t timeout)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    void *old_userdata;

    assert(session != NULL && timeout >= 1);

    node_p = (abcdk_asynctcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    abcdk_asynctcp_set_timeout(node_p,timeout * 1000);
}

void abcdk_srpc_destroy(abcdk_srpc_t **ctx)
{
    abcdk_srpc_t *ctx_p;

    if(!ctx ||!*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_asynctcp_stop(&ctx_p->io_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_srpc_t *abcdk_srpc_create(int max,int cpu)
{
    abcdk_srpc_t *ctx;

    assert(max > 0);

    ctx = abcdk_heap_alloc(sizeof(abcdk_srpc_t));
    if(!ctx)
        return NULL;

    ctx->io_ctx = abcdk_asynctcp_start(max, cpu);
    if (!ctx->io_ctx)
        goto ERR;

    return ctx;
ERR:

    abcdk_srpc_destroy(&ctx);

    return NULL;
}

static void _abcdk_srpc_prepare_cb(abcdk_asynctcp_node_t **node, abcdk_asynctcp_node_t *listen)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_srpc_node_t *listen_ctx_p, *node_ctx_p;
    abcdk_srpc_config_t *cfg_p;
    int chk;

    listen_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(listen);

    listen_ctx_p->cfg.prepare_cb(listen_ctx_p->cfg.opaque,(abcdk_srpc_session_t **)&node_p,(abcdk_srpc_session_t *)listen);
    if (!node_p)
        return;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    node_ctx_p->cfg = listen_ctx_p->cfg;
    node_ctx_p->flag = 1;

    cfg_p = &node_ctx_p->cfg;

    if(cfg_p->ssl_scheme == ABCDK_SRPC_SSL_SCHEME_OPENSSL)
    {
        chk = abcdk_asynctcp_upgrade2openssl(node_p,listen_ctx_p->openssl_ctx);
        if(chk != 0)
            abcdk_asynctcp_unref(&node_p);
    }
    else if(cfg_p->ssl_scheme == ABCDK_SRPC_SSL_SCHEME_EASYSSL)
    {
        node_ctx_p->easyssl_ctx = abcdk_easyssl_create_from_file(cfg_p->easyssl_key_file,ABCDK_EASYSSL_SCHEME_ENIGMA,ABCDK_CLAMP(cfg_p->easyssl_salt_size,0,256));
        if(!node_ctx_p->easyssl_ctx)
            abcdk_asynctcp_unref(&node_p);
        else
            abcdk_asynctcp_upgrade2easyssl(node_p,node_ctx_p->easyssl_ctx);
    }

    /*准备完毕，返回。*/
    *node = node_p;
}

static void _abcdk_srpc_event_accept(abcdk_asynctcp_node_t *node, int *result)
{
    abcdk_srpc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node);

    abcdk_asynctcp_get_sockaddr_str(node, node_ctx_p->local_addr, node_ctx_p->remote_addr);

    /*默认：允许。*/
    *result = 0;

    if(node_ctx_p->cfg.accept_cb)
        node_ctx_p->cfg.accept_cb(node_ctx_p->cfg.opaque, (abcdk_srpc_session_t*)node, result);
    
    if(*result != 0)
        abcdk_trace_output(LOG_INFO, "禁止客户端('%s')连接到本机('%s')。", node_ctx_p->remote_addr, node_ctx_p->local_addr);
}

static int _abcdk_srpc_event_connect_check_cert(abcdk_asynctcp_node_t *node)
{
    abcdk_srpc_node_t *node_ctx_p;
    SSL *ssl_p;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node);

    if (!node_ctx_p->cfg.openssl_no_check_cert)
        return 0;

#ifdef HEADER_SSL_H

    ssl_p = abcdk_asynctcp_openssl_ctx(node);

    /*检测SSL环境验证结果。*/
    chk = SSL_get_verify_result(ssl_p);
    if (chk != X509_V_OK)
        return -1;

#endif // HEADER_SSL_H


    return 0;
}

static void _abcdk_srpc_event_connect(abcdk_asynctcp_node_t *node)
{
    abcdk_srpc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node);

    /*设置超时。*/
    abcdk_asynctcp_set_timeout(node, 180 * 1000);

    if(!node_ctx_p->remote_addr[0])
        abcdk_asynctcp_get_sockaddr_str(node,NULL,node_ctx_p->remote_addr);
    if(!node_ctx_p->local_addr[0])
        abcdk_asynctcp_get_sockaddr_str(node,node_ctx_p->local_addr,NULL);

    /*检查证书。*/
    chk = _abcdk_srpc_event_connect_check_cert(node);
    if(chk != 0)
    {
        abcdk_trace_output(LOG_INFO, "验证('%s')的证书失败，证书已过期或未生效。", node_ctx_p->remote_addr);

        /*修改超时，使用超时检测器关闭。*/
        abcdk_asynctcp_set_timeout(node, 1);
        return;
    }

END:

    abcdk_trace_output(LOG_INFO, "本机('%s')与%s('%s')的连接已经建立。",
                       node_ctx_p->local_addr,
                       (node_ctx_p->flag == 1 ? "客户端" : "服务端"),
                       node_ctx_p->remote_addr);

    if(node_ctx_p->cfg.ready_cb)
        node_ctx_p->cfg.ready_cb(node_ctx_p->cfg.opaque, (abcdk_srpc_session_t*)node);

    /*已连接到远端，注册读写事件。*/
    abcdk_asynctcp_recv_watch(node);
    abcdk_asynctcp_send_watch(node);
}

static void _abcdk_srpc_event_output(abcdk_asynctcp_node_t *node)
{
    abcdk_srpc_node_t *node_ctx_p;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node);

    if(node_ctx_p->cfg.output_cb)
        node_ctx_p->cfg.output_cb(node_ctx_p->cfg.opaque, (abcdk_srpc_session_t*)node);
}

static void _abcdk_srpc_event_close(abcdk_asynctcp_node_t *node)
{
    abcdk_srpc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node);

    if(!node_ctx_p->remote_addr[0])
        abcdk_asynctcp_get_sockaddr_str(node,NULL,node_ctx_p->remote_addr);
    if(!node_ctx_p->local_addr[0])
        abcdk_asynctcp_get_sockaddr_str(node,node_ctx_p->local_addr,NULL);

    if(!node_ctx_p->flag)
        abcdk_trace_output(LOG_INFO, "本机('%s')与%s('%s')的连接已经断开。", node_ctx_p->local_addr,(node_ctx_p->flag == 1 ? "客户端" : "服务端"), node_ctx_p->remote_addr);
    else 
        abcdk_trace_output(LOG_INFO, "本机('%s')与%s('%s')的连接已经断开。", node_ctx_p->local_addr,(node_ctx_p->flag == 1 ? "客户端" : "服务端"), node_ctx_p->remote_addr);

    /*如果连接关闭则一定要取消等待的事务，否则可能会造成应用层阻塞。*/
    if(node_ctx_p->flag)
        abcdk_waiter_cancel(node_ctx_p->req_waiter);

    if(node_ctx_p->cfg.close_cb)
        node_ctx_p->cfg.close_cb(node_ctx_p->cfg.opaque,(abcdk_srpc_session_t*)node);
}

static void _abcdk_srpc_event_cb(abcdk_asynctcp_node_t *node, uint32_t event, int *result)
{
    abcdk_srpc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node);

    if (event == ABCDK_ASYNCTCP_EVENT_ACCEPT)
    {
        _abcdk_srpc_event_accept(node,result);
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_CONNECT)
    {
        _abcdk_srpc_event_connect(node);
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_INPUT)
    {
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_OUTPUT)
    {
        _abcdk_srpc_event_output(node);
    }
    else if (event == ABCDK_ASYNCTCP_EVENT_CLOSE || event == ABCDK_ASYNCTCP_EVENT_INTERRUPT)
    {
        _abcdk_srpc_event_close(node);
    }
}

static void _abcdk_srpc_process(abcdk_asynctcp_node_t *node)
{
    abcdk_srpc_node_t *node_ctx_p;
    const void *req_data;
    size_t req_size;
    uint32_t len;
    uint8_t cmd;
    uint64_t mid;
    abcdk_object_t *cargo;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node);

    req_data = abcdk_receiver_data(node_ctx_p->req_data, 0);
    req_size = abcdk_receiver_length(node_ctx_p->req_data);

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
            node_ctx_p->cfg.request_cb(node_ctx_p->cfg.opaque,(abcdk_srpc_session_t*)node,mid,ABCDK_PTR2VPTR(req_data, 13), req_size - 13);
    }
}

static void _abcdk_srpc_request_cb(abcdk_asynctcp_node_t *node, const void *data, size_t size, size_t *remain)
{
    abcdk_srpc_node_t *node_ctx_p;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node);

    /*默认没有剩余数据。*/
    *remain = 0;

    if (!node_ctx_p->req_data)
    {
        node_ctx_p->req_data = abcdk_receiver_alloc(ABCDK_RECEIVER_PROTO_SMB, 16*1024*1024, NULL);
    }

    if (!node_ctx_p->req_data)
        goto ERR;

    chk = abcdk_receiver_append(node_ctx_p->req_data, data, size, remain);
    if (chk < 0)
        goto ERR;
    else if (chk == 0) /*数据包不完整，继续接收。*/
        return;

    _abcdk_srpc_process(node);

    /*一定要回收。*/
    abcdk_receiver_unref(&node_ctx_p->req_data);

    /*No Error.*/
    return;

ERR:

    abcdk_asynctcp_set_timeout(node, 1);
}

static int _abcdk_srpc_ssl_init(abcdk_srpc_session_t *session,int server)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    abcdk_srpc_config_t *cfg_p;
    int chk;

    node_p = (abcdk_asynctcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    cfg_p = &node_ctx_p->cfg;

    if (cfg_p->ssl_scheme == ABCDK_SRPC_SSL_SCHEME_OPENSSL)
    {
#ifdef HEADER_SSL_H
        node_ctx_p->openssl_ctx = abcdk_openssl_ssl_ctx_alloc_load(server, cfg_p->openssl_ca_file, cfg_p->openssl_ca_path, cfg_p->openssl_cert_file, cfg_p->openssl_key_file, NULL);
#endif // HEADER_SSL_H
        if (!node_ctx_p->openssl_ctx)
        {
            abcdk_trace_output(LOG_WARNING, "加载证书或私钥失败，无法创建SSL环境。");
            return -2;
        }

        /*仅客户端需要。*/
        if(!server)
        {
            chk = abcdk_asynctcp_upgrade2openssl(node_p,node_ctx_p->openssl_ctx);
            if(chk != 0)
                return -3;
        }
        
    }
    else if (cfg_p->ssl_scheme == ABCDK_SRPC_SSL_SCHEME_EASYSSL)
    {
        node_ctx_p->easyssl_ctx = abcdk_easyssl_create_from_file(cfg_p->easyssl_key_file,ABCDK_EASYSSL_SCHEME_ENIGMA,ABCDK_CLAMP(cfg_p->easyssl_salt_size,0,256));
        if (!node_ctx_p->easyssl_ctx)
        {
            abcdk_trace_output(LOG_WARNING, "加载私钥失败，无法创建SSL环境。");
            return -2;
        }

        /*仅客户端需要。*/
        if(!server)
        {
            chk = abcdk_asynctcp_upgrade2easyssl(node_p,node_ctx_p->easyssl_ctx);
            if (chk != 0)
                return -3;
        }
    }

    return 0;
}

int abcdk_srpc_listen(abcdk_srpc_session_t *session,abcdk_sockaddr_t *addr,abcdk_srpc_config_t *cfg)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    abcdk_asynctcp_callback_t cb = {0};
    int chk;

    assert(session != NULL && addr != NULL && cfg != NULL);
    assert(cfg->prepare_cb != NULL && cfg->request_cb != NULL);

    node_p = (abcdk_asynctcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    node_ctx_p->cfg = *cfg;
    node_ctx_p->flag = 0;

    /*初始化安全模式。*/
    chk = _abcdk_srpc_ssl_init(session,1);
    if(chk != 0)
        return -2;

    cb.prepare_cb = _abcdk_srpc_prepare_cb;
    cb.event_cb = _abcdk_srpc_event_cb;
    cb.request_cb = _abcdk_srpc_request_cb;

    chk = abcdk_asynctcp_listen(node_p,addr,&cb);
    if(chk != 0)
        return -1;

    return 0;
}

int abcdk_srpc_connect(abcdk_srpc_session_t *session,abcdk_sockaddr_t *addr,abcdk_srpc_config_t *cfg)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    abcdk_asynctcp_callback_t cb = {0};
    int chk;

    assert(session != NULL && addr != NULL && cfg != NULL);
    assert(cfg->prepare_cb != NULL && cfg->request_cb != NULL);

    node_p = (abcdk_asynctcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    node_ctx_p->cfg = *cfg;
    node_ctx_p->flag = 2;

    /*初始化安全模式。*/
    chk = _abcdk_srpc_ssl_init(session,0);
    if(chk != 0)
        return -2;

    cb.prepare_cb = _abcdk_srpc_prepare_cb;
    cb.event_cb = _abcdk_srpc_event_cb;
    cb.request_cb = _abcdk_srpc_request_cb;

    chk = abcdk_asynctcp_connect(node_p,addr,&cb);
    if(chk != 0)
        return -1;

    return 0;
}

static int _abcdk_srpc_post(abcdk_asynctcp_node_t *node,  uint8_t cmd, uint64_t mid, const void *data, size_t size)
{
    abcdk_srpc_node_t *node_ctx_p;
    abcdk_object_t *msg;
    int chk;

    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node);

    /*
     * |Length  |CMD    |MID     |Data    |
     * |4 Bytes |1 Byte |8 Bytes |N Bytes |
     *
     * Length： 不包含自身。
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

    chk = abcdk_asynctcp_post(node,msg);
    if(chk == 0)
        return 0;

    /*如果发送失败则删除消息，避免发生内存泄漏。*/
    abcdk_object_unref(&msg);
    return -2;
}

int abcdk_srpc_request(abcdk_srpc_session_t *session, const void *req, size_t req_size, abcdk_object_t **rsp)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    uint64_t mid;
    int chk;

    assert(session != NULL && req != NULL && req_size > 0);

    node_p = (abcdk_asynctcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    mid = abcdk_atomic_fetch_and_add(&node_ctx_p->mid_next, 1);

    if (rsp)
    {
        chk = abcdk_waiter_register(node_ctx_p->req_waiter, mid);
        if (chk != 0)
            return -1;
    }

    chk = _abcdk_srpc_post(node_p,2,mid,req,req_size);
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
    abcdk_asynctcp_node_t *node_p;
    abcdk_srpc_node_t *node_ctx_p;
    int chk;

    assert(session != NULL && data != NULL && size >0);

    node_p = (abcdk_asynctcp_node_t*)session;
    node_ctx_p = (abcdk_srpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    chk = _abcdk_srpc_post(node_p,1,mid,data,size);
    if(chk != 0)
        return -1;

    return 0;
}