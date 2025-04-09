/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/rtsp/server.h"
#include "abcdk/rtsp/rtsp.h"
#include "server.hxx"
#include "server_auth.hxx"

__BEGIN_DECLS

#ifdef _RTSP_SERVER_HH

/*简单的RTSP服务。*/
struct _abcdk_rtsp_server
{
    /*工作标志。0 运行，!0 退出。*/
    volatile char worker_flag;

    /**工作线程。 */
    abcdk_thread_t worker_thread;

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

    ABCDK_ASSERT(ctx_p->worker_flag,"停止后才能销毁。");

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

abcdk_rtsp_server_t *abcdk_rtsp_server_create(uint16_t port, const char *realm)
{
    abcdk_rtsp_server_t *ctx;

    assert(port > 0 && port < 65536);

    ctx = (abcdk_rtsp_server_t*)abcdk_heap_alloc(sizeof(abcdk_rtsp_server_t));
    if(!ctx)
        return NULL;

    /*标记为退出。*/
    ctx->worker_flag = 1;

    ctx->l5_scheduler_ctx = BasicTaskScheduler::createNew();
    if(!ctx->l5_scheduler_ctx)
        goto ERR;

    ctx->l5_env_ctx = BasicUsageEnvironment::createNew(*ctx->l5_scheduler_ctx);
    if(!ctx->l5_env_ctx)
        goto ERR;

    ctx->l5_auth_ctx = new abcdk::rtsp_server::auth(realm);
    if(!ctx->l5_auth_ctx)
        goto ERR;

    ctx->l5_auth_ctx->addUserRecord("aaaa","bbbb");

    ctx->l5_server_ctx = abcdk::rtsp::server::createNew(*ctx->l5_env_ctx, port, ctx->l5_auth_ctx);
    if(!ctx->l5_server_ctx)
        goto ERR;

    return ctx;
ERR:

    abcdk_rtsp_server_destroy(&ctx);
    return NULL;
}

int abcdk_rtsp_server_enable_tls(abcdk_rtsp_server_t *ctx,const char *cert,const char *key)
{
    assert(ctx != NULL && cert != NULL && key != NULL);

    ABCDK_ASSERT(!ctx->worker_flag,"服务已经启动，不允许修改基础配置。");

    ctx->l5_server_ctx->setTLSState(cert,key);

    return 0;
}


static void *_abcdk_rtsp_server_worker_thread_routine(void *opaque)
{
    abcdk_rtsp_server_t *ctx = (abcdk_rtsp_server_t *)opaque;

    ctx->l5_env_ctx->taskScheduler().doEventLoop(&ctx->worker_flag);

    return NULL;
}


void abcdk_rtsp_server_stop(abcdk_rtsp_server_t *ctx)
{
    assert(ctx != NULL);

    ctx->worker_flag = 1;
    abcdk_thread_join(&ctx->worker_thread);

    ctx->l5_server_ctx->remove_media_all();
}

int abcdk_rtsp_server_start(abcdk_rtsp_server_t *ctx)
{
    int chk;

//     ctx->worker_flag = 0;
//    ctx->l5_env_ctx->taskScheduler().doEventLoop(&ctx->worker_flag);

    assert(ctx != NULL);

    ctx->worker_flag = 0;
    ctx->worker_thread.routine = _abcdk_rtsp_server_worker_thread_routine;
    ctx->worker_thread.opaque = ctx;

    chk = abcdk_thread_create(&ctx->worker_thread, 1);
    if (chk != 0)
        return -1;

    return 0;
}

void abcdk_rtsp_server_remove_media(abcdk_rtsp_server_t *ctx, int media)
{
    assert(ctx != NULL && media > 0);

    ctx->l5_server_ctx->remove_media(media);
}


int abcdk_rtsp_server_media_play(abcdk_rtsp_server_t *ctx, int media)
{
    int chk;

    assert(ctx != NULL && media > 0);

    chk = ctx->l5_server_ctx->media_play(media);
    if(chk != 0)
        return -1;

    return 0;
}


int abcdk_rtsp_server_create_media(abcdk_rtsp_server_t *ctx, const char *name, const char *info, const char *desc)
{
    int chk;

    assert(ctx != NULL && name != NULL);

    chk = ctx->l5_server_ctx->create_media(name,info,desc);
    if(chk <= 0)
        return -1;

    return chk;
}

int abcdk_rtsp_server_media_add_stream(abcdk_rtsp_server_t *ctx, int media, int codec, abcdk_object_t *extdata, int cache)
{
    int chk;

    assert(ctx != NULL && media > 0);

    chk = ctx->l5_server_ctx->media_add_stream(media, codec, extdata, cache);
    if (chk <= 0)
        return -1;

    return chk;
}

int abcdk_rtsp_server_media_append_stream(abcdk_rtsp_server_t *ctx, int media, int stream, const void *data, size_t size, int64_t dts, int64_t pts, int64_t dur)
{
    int chk;

    assert(ctx != NULL && media > 0 && stream > 0);

    chk = ctx->l5_server_ctx->media_append_stream(media,stream,data,size,dts,pts,dur);
    if(chk != 0)
        return -1;

    return 0;
}


#else //_RTSP_SERVER_HH



#endif //_RTSP_SERVER_HH


__END_DECLS