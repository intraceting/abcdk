/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/rtsp/server.h"
#include "abcdk/rtsp/live555.h"
#include "server.hxx"
#include "server_auth.hxx"

__BEGIN_DECLS

#ifdef _RTSP_SERVER_HH

/*简单的RTSP服务。*/
struct _abcdk_rtsp_server
{
    /*退出标志。0 运行，!0 退出。*/
    volatile int8_t exit_flag;

    TaskScheduler* l5_scheduler_ctx;
    UsageEnvironment* l5_env_ctx;
    abcdk::rtsp::server* l5_server_ctx;
    abcdk::rtsp_server::auth *l5_auth_ctx;

};//abcdk_rtsp_server_t;


void abcdk_rtsp_server_destroy(abcdk_rtsp_server_t **ctx)
{
    abcdk_rtsp_server_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p->l5_server_ctx)
    {
        abcdk::rtsp::server::deleteOld(&ctx_p->l5_server_ctx);
        ctx_p->l5_server_ctx = NULL;
    }

    if (ctx_p->l5_auth_ctx)
    {
        delete ctx_p->l5_auth_ctx;
        ctx_p->l5_auth_ctx = NULL;
    }

    if (ctx_p->l5_env_ctx)
    {
        ctx_p->l5_env_ctx->reclaim();
        ctx_p->l5_env_ctx = NULL;
    }

    if (ctx_p->l5_scheduler_ctx)
    {
        delete ctx_p->l5_scheduler_ctx;
        ctx_p->l5_scheduler_ctx = NULL;
    }

    abcdk_heap_free(ctx_p);

}

abcdk_rtsp_server_t *abcdk_rtsp_server_create(uint16_t port, char const *realm)
{
    abcdk_rtsp_server_t *ctx;

    assert(port > 0 && port < 65536);

    ctx = (abcdk_rtsp_server_t*)abcdk_heap_alloc(sizeof(abcdk_rtsp_server_t*));
    if(!ctx)
        return NULL;

    ctx->l5_scheduler_ctx = BasicTaskScheduler::createNew();
    if(!ctx->l5_scheduler_ctx)
        goto ERR;

    ctx->l5_env_ctx = BasicUsageEnvironment::createNew(*ctx->l5_scheduler_ctx);
    if(!ctx->l5_env_ctx)
        goto ERR;

    ctx->l5_auth_ctx = new abcdk::rtsp_server::auth(realm);
    if(!ctx->l5_auth_ctx)
        goto ERR;

    ctx->l5_server_ctx = abcdk::rtsp::server::createNew(*ctx->l5_env_ctx, port, ctx->l5_auth_ctx);
    if(!ctx->l5_env_ctx)
        goto ERR;

    return ctx;
ERR:

    abcdk_rtsp_server_destroy(&ctx);
    return NULL;
}


#else //_RTSP_SERVER_HH



#endif //_RTSP_SERVER_HH


__END_DECLS