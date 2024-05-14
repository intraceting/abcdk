/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <locale.h>
#include "entry.h"


typedef struct _abcdkrpc
{
    int errcode;
    abcdk_option_t *args;

    /*CA证书。*/
    const char *ca_file;

    /*CA证书目录。*/
    const char *ca_path;

    /*服务器证书。*/
    const char *cert_file;

    /*服务器证书私钥。*/
    const char *key_file;

    /*通讯环境。*/
    abcdk_rpc_t *io_ctx;
    abcdk_rpc_session_t *listen_p;
    abcdk_rpc_session_t *uplink_p[10];

}abcdkrpc_t;

static void _abcdk_test_rpc_prepare_cb(void *opaque, abcdk_rpc_session_t **session, abcdk_rpc_session_t *listen)
{
    *session = abcdk_rpc_alloc(((abcdkrpc_t *)opaque)->io_ctx);
}

static void _abcdk_test_rpc_accept_cb(void *opaque, abcdk_rpc_session_t *session, int *result)
{
    *result = 0;
}

static void _abcdk_test_rpc_ready_cb(void *opaque, abcdk_rpc_session_t *session)
{
    abcdk_rpc_set_timeout(session, 30);
}

static void _abcdk_test_rpc_close_cb(void *opaque, abcdk_rpc_session_t *session)
{
}

static void _abcdk_test_rpc_request_cb(void *opaque, abcdk_rpc_session_t *session, uint64_t mid, const void *data, size_t size)
{

}

static int _abcdkhttpd_start_listen(abcdkrpc_t *ctx)
{
    const char *listen;
    abcdk_sockaddr_t listen_addr = {0};
    abcdk_rpc_config_t cfg = {0};
    abcdk_rpc_session_t *listen_p;
    int chk;

    listen = abcdk_option_get(ctx->args, "--listen", 0, NULL);

    /*未启用。*/
    if(!listen)
        return 0;

    chk = abcdk_sockaddr_from_string(&listen_addr, listen, 0);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "监听地址'%s'无法识别。", listen);
        return -1;
    }

    cfg.opaque = ctx;
    cfg.prepare_cb = _abcdk_test_rpc_prepare_cb;
    cfg.accept_cb = _abcdk_test_rpc_accept_cb;
    cfg.ready_cb = _abcdk_test_rpc_ready_cb;
    cfg.close_cb = _abcdk_test_rpc_close_cb;
    cfg.request_cb = _abcdk_test_rpc_request_cb;
 
    cfg.ca_file = ctx->ca_file;
    cfg.ca_path = ctx->ca_path;
    cfg.cert_file = ctx->cert_file;
    cfg.key_file = ctx->key_file;

    listen_p = ctx->listen_p = abcdk_rpc_alloc(ctx->io_ctx);
    if (!listen_p)
    {
        abcdk_trace_output(LOG_ERR, "内部错误。");
        return -2;
    }

    chk = abcdk_rpc_listen(listen_p,&listen_addr,&cfg);
    if(chk == 0)
        return 0;

    return -3;

}

static int _abcdkhttpd_connect_uplink(abcdkrpc_t *ctx,int idx)
{
    const char *uplink;
    abcdk_sockaddr_t uplink_addr = {0};
    abcdk_rpc_config_t cfg = {0};
    abcdk_rpc_session_t *uplink_p;
    int chk;

    uplink = abcdk_option_get(ctx->args, "--up-link", 0, NULL);

    if(!uplink)
        return -1;

    chk = abcdk_sockaddr_from_string(&uplink_addr, uplink, 1);
    if (chk != 0)
    {
        abcdk_trace_output(LOG_ERR, "上级地址'%s'无法识别。", uplink);
        return -1;
    }

    cfg.opaque = ctx;
    cfg.prepare_cb = _abcdk_test_rpc_prepare_cb;
    cfg.accept_cb = _abcdk_test_rpc_accept_cb;
    cfg.ready_cb = _abcdk_test_rpc_ready_cb;
    cfg.close_cb = _abcdk_test_rpc_close_cb;
    cfg.request_cb = _abcdk_test_rpc_request_cb;
 
    cfg.ca_file = ctx->ca_file;
    cfg.ca_path = ctx->ca_path;
    cfg.cert_file = ctx->cert_file;
    cfg.key_file = ctx->key_file;

    uplink_p = ctx->uplink_p[idx] = abcdk_rpc_alloc(ctx->io_ctx);
    if (!uplink_p)
    {
        abcdk_trace_output(LOG_ERR, "内部错误。");
        return -2;
    }

    chk = abcdk_rpc_connect(uplink_p,&uplink_addr,&cfg);
    if(chk == 0)
        return 0;

    return -3;

}

static void _abcdk_test_work(abcdkrpc_t *ctx)
{
    int chk;

    ctx->ca_file = abcdk_option_get(ctx->args, "--ca-file", 0, NULL);
    ctx->ca_path = abcdk_option_get(ctx->args, "--ca-path", 0, NULL);
    ctx->cert_file = abcdk_option_get(ctx->args, "--cert-file", 0, NULL);
    ctx->key_file = abcdk_option_get(ctx->args, "--key-file", 0, NULL);

    ctx->io_ctx = abcdk_rpc_create(256, -1);
    if (!ctx->io_ctx)
    {
        abcdk_trace_output(LOG_WARNING, "内存错误。\n");
        goto final;
    }

    chk = _abcdkhttpd_start_listen(ctx);
    if(chk != 0)
        goto final;



final:

    abcdk_rpc_destroy(&ctx->io_ctx);
    abcdk_rpc_unref(&ctx->listen_p);
}

int abcdk_test_rpc(abcdk_option_t *args)
{
    abcdkrpc_t ctx = {0};

    ctx.args = args;

    _abcdk_test_work(&ctx);

}