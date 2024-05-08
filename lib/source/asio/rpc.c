/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/asio/rpc.h"

/**简单的RPC服务。*/
struct _abcdk_rpc
{
    /*配置。*/
    abcdk_rpc_config_t cfg;

    /*通讯IO。*/
    abcdk_asynctcp_t *io_ctx;
};//abcdk_rpc_t


typedef struct _abcdk_rpc_node
{
    /*父级。*/
    abcdk_rpc_t *father;

    /*SSL环境。*/
    SSL_CTX *ssl_ctx;

    /*远程地址。*/
    char remote_addr[NAME_MAX];

    /*标志。0 监听，1 服务端，2 客户端。*/
    int flag;

    /*请求数据。*/
    abcdk_receiver_t *req_data;

    /*请求服务员。*/
    abcdk_waiter_t *req_waiter;

    /*用户环境指针。*/
    void *userdata;

} abcdk_rpc_node_t;


void abcdk_rpc_session_unref(abcdk_rpc_session_t **session)
{
    abcdk_asynctcp_unref((abcdk_asynctcp_node_t**)session);
}


abcdk_rpc_session_t *abcdk_rpc_session_refer(abcdk_rpc_session_t *src)
{
    return (abcdk_rpc_session_t*)abcdk_asynctcp_refer((abcdk_asynctcp_node_t*)src);
}

static void _abcdk_rpc_node_destroy_cb(void *userdata)
{
    abcdk_rpc_node_t *ctx;

    if (!userdata)
        return;

    ctx = (abcdk_rpc_node_t *)userdata;

#ifdef HEADER_SSL_H
    abcdk_openssl_ssl_ctx_free(&ctx->ssl_ctx);
#endif //HEADER_SSL_H

    abcdk_receiver_unref(&ctx->req_data);
    abcdk_waiter_free(&ctx->req_waiter);

}

static void _abcdk_rpc_node_waiter_msg_destroy_cb(void *msg)
{

}

abcdk_rpc_session_t *abcdk_rpc_session_alloc(abcdk_rpc_t *ctx)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_rpc_node_t *node_ctx_p;

    assert(ctx != NULL);

    node_p = abcdk_asynctcp_alloc(ctx->io_ctx, sizeof(abcdk_rpc_node_t), _abcdk_rpc_node_destroy_cb);
    if (!node_p)
        return NULL;

    node_ctx_p = (abcdk_rpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    node_ctx_p->father = ctx;
    node_ctx_p->req_waiter = abcdk_waiter_alloc(_abcdk_rpc_node_waiter_msg_destroy_cb);

    return (abcdk_rpc_session_t*)node_p;
}

void *abcdk_rpc_session_get_userdata(abcdk_rpc_session_t *session)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_rpc_node_t *node_ctx_p;

    assert(session != NULL);

    node_p = (abcdk_asynctcp_node_t*)session;
    node_ctx_p = (abcdk_rpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    return node_ctx_p->userdata;
}

void *abcdk_rpc_session_set_userdata(abcdk_rpc_session_t *session,void *userdata)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_rpc_node_t *node_ctx_p;
    void *old_userdata;

    assert(session != NULL);

    node_p = (abcdk_asynctcp_node_t*)session;
    node_ctx_p = (abcdk_rpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    old_userdata = node_ctx_p->userdata;
    node_ctx_p->userdata = userdata;
    
    return old_userdata;
}

const char *abcdk_rpc_session_get_address(abcdk_rpc_session_t *session, int remote)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_rpc_node_t *node_ctx_p;
    void *old_userdata;

    assert(session != NULL);

    node_p = (abcdk_asynctcp_node_t *)session;
    node_ctx_p = (abcdk_rpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    if (remote)
        return node_ctx_p->remote_addr;
    else
        return "";
}

void abcdk_rpc_session_set_timeout(abcdk_rpc_session_t *session,time_t timeout)
{
    abcdk_asynctcp_node_t *node_p;
    abcdk_rpc_node_t *node_ctx_p;
    void *old_userdata;

    assert(session != NULL && timeout >= 1);

    node_p = (abcdk_asynctcp_node_t*)session;
    node_ctx_p = (abcdk_rpc_node_t *)abcdk_asynctcp_get_userdata(node_p);

    abcdk_asynctcp_set_timeout(node_p,timeout * 1000);
}

void abcdk_rpc_destroy(abcdk_rpc_t **ctx)
{
    abcdk_rpc_t *ctx_p;

    if(!ctx ||!*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_asynctcp_stop(&ctx_p->io_ctx);
    abcdk_heap_free(ctx_p);
}

abcdk_rpc_t *abcdk_rpc_create(int max,int cpu)
{
    abcdk_rpc_t *ctx;

    assert(max > 0);

    ctx = abcdk_heap_alloc(sizeof(abcdk_rpc_t));
    if(!ctx)
        return NULL;

    ctx->io_ctx = abcdk_asynctcp_start(max, cpu);
    if (!ctx->io_ctx)
        goto ERR;

    return ctx;
ERR:

    abcdk_rpc_destroy(&ctx);

    return NULL;
}