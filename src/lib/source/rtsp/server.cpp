/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/rtsp/server.h"
#include "abcdk/rtsp/rtsp.h"
#include "server_env.hxx"
#include "server_auth.hxx"
#include "server.hxx"

__BEGIN_DECLS

#ifdef _RTSP_SERVER_HH

#define ABCDK_RTSP_SERVER_REALM "ABCDK MediaServer"

/*简单的RTSP服务。*/
struct _abcdk_rtsp_server
{
    /*工作标志。0 运行，!0 退出。*/
    volatile char worker_flag;

    /**工作线程。 */
    abcdk_thread_t worker_thread;

    TaskScheduler *l5_scheduler_ctx;
    abcdk::rtsp_server::env *l5_env_ctx;
    abcdk::rtsp::server *l5_server_ctx;

}; // abcdk_rtsp_server_t;

void abcdk_rtsp_server_destroy(abcdk_rtsp_server_t **ctx)
{
    abcdk_rtsp_server_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    ABCDK_ASSERT(ctx_p->worker_flag, TT("服务停止后才能销毁。"));

    if (ctx_p->l5_server_ctx)
    {
        abcdk::rtsp::server::deleteOld(&ctx_p->l5_server_ctx);
        ctx_p->l5_server_ctx = NULL;
    }

    if (ctx_p->l5_env_ctx)
    {
        abcdk::rtsp_server::env::deleteOld(&ctx_p->l5_env_ctx);
        ctx_p->l5_env_ctx = NULL;
    }

    if (ctx_p->l5_scheduler_ctx)
    {
        delete ctx_p->l5_scheduler_ctx;
        ctx_p->l5_scheduler_ctx = NULL;
    }

    abcdk_heap_free(ctx_p);
}

abcdk_rtsp_server_t *abcdk_rtsp_server_create(uint16_t port, int flag)
{
    abcdk_rtsp_server_t *ctx;

    assert(port > 0 && port < 65536);

    ctx = (abcdk_rtsp_server_t *)abcdk_heap_alloc(sizeof(abcdk_rtsp_server_t));
    if (!ctx)
        return NULL;

    /*标记为退出。*/
    ctx->worker_flag = 1;

    ctx->l5_scheduler_ctx = BasicTaskScheduler::createNew();
    if (!ctx->l5_scheduler_ctx)
        goto ERR;

    ctx->l5_env_ctx = abcdk::rtsp_server::env::createNew(*ctx->l5_scheduler_ctx);
    if (!ctx->l5_env_ctx)
        goto ERR;

    ctx->l5_server_ctx = abcdk::rtsp::server::createNew(*ctx->l5_env_ctx, port, flag);
    if (!ctx->l5_server_ctx)
        goto ERR;

    return ctx;
ERR:

    abcdk_rtsp_server_destroy(&ctx);
    return NULL;
}

int abcdk_rtsp_server_set_auth(abcdk_rtsp_server_t *ctx, const char *realm)
{
    int chk;

    assert(ctx != NULL && realm != NULL);

    ABCDK_ASSERT(ctx->worker_flag, TT("服务已经启动，禁止修改基础配置。"));

    chk = ctx->l5_server_ctx->set_auth(realm ? realm : ABCDK_RTSP_SERVER_REALM);
    if(chk != 0)
        return -1;

    return 0;
}

int abcdk_rtsp_server_set_tls(abcdk_rtsp_server_t *ctx, const char *cert, const char *key, int enable_srtp, int encrypt_srtp)
{
    int chk;

    assert(ctx != NULL && cert != NULL && key != NULL);

    ABCDK_ASSERT(ctx->worker_flag, TT("服务已经启动，禁止修改基础配置。"));

    chk = ctx->l5_server_ctx->set_tls(cert, key, enable_srtp, encrypt_srtp);
    if (chk != 0)
        return -1;

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
}

int abcdk_rtsp_server_start(abcdk_rtsp_server_t *ctx)
{
    int chk;

    assert(ctx != NULL);

    ctx->worker_flag = 0;
    ctx->worker_thread.routine = _abcdk_rtsp_server_worker_thread_routine;
    ctx->worker_thread.opaque = ctx;

    chk = abcdk_thread_create(&ctx->worker_thread, 1);
    if (chk != 0)
        return -1;

    return 0;
}

void abcdk_rtsp_server_remove_user(abcdk_rtsp_server_t *ctx, const char *username)
{
    assert(ctx != NULL && username != NULL);

    ABCDK_ASSERT(!ctx->worker_flag, TT("服务尚未启动，禁止修改运行配置。"));

    ctx->l5_server_ctx->remove_user(username);
}

int abcdk_rtsp_server_add_user(abcdk_rtsp_server_t *ctx, const char *username, const char *password)
{
    int chk;

    assert(ctx != NULL && username != NULL && password != NULL);

    ABCDK_ASSERT(!ctx->worker_flag, TT("服务尚未启动，禁止修改运行配置。"));

    chk = ctx->l5_server_ctx->add_user(username, password);
    if(chk != 0)
        return -1;

    return 0;
}

void abcdk_rtsp_server_remove_media(abcdk_rtsp_server_t *ctx, int media)
{
    assert(ctx != NULL && media > 0);

    ABCDK_ASSERT(!ctx->worker_flag, TT("服务尚未启动，禁止修改运行配置。"));

    ctx->l5_server_ctx->remove_media(media);
}

int abcdk_rtsp_server_play_media(abcdk_rtsp_server_t *ctx, int media)
{
    int chk;

    assert(ctx != NULL && media > 0);

    ABCDK_ASSERT(!ctx->worker_flag, TT("服务尚未启动，禁止修改运行配置。"));

    chk = ctx->l5_server_ctx->play_media(media);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_rtsp_server_create_media(abcdk_rtsp_server_t *ctx, const char *name, const char *title, const char *comment)
{
    int chk;

    assert(ctx != NULL && name != NULL);

    ABCDK_ASSERT(!ctx->worker_flag, TT("服务尚未启动，禁止修改运行配置。"));

    chk = ctx->l5_server_ctx->create_media(name, (comment ? comment : ABCDK_RTSP_SERVER_REALM), (title ? title : ABCDK_RTSP_SERVER_REALM));
    if (chk <= 0)
        return -1;

    return chk;
}

int abcdk_rtsp_server_add_stream(abcdk_rtsp_server_t *ctx, int media, int codec, abcdk_object_t *extdata, uint32_t bitrate, int cache)
{
    int chk;

    assert(ctx != NULL && media > 0 && codec > ABCDK_RTSP_CODEC_NONE && extdata != NULL && bitrate > 0 && cache >= 2);

    ABCDK_ASSERT(!ctx->worker_flag, TT("服务尚未启动，禁止修改运行配置。"));

    chk = ctx->l5_server_ctx->add_stream(media, codec, extdata, bitrate, cache);
    if (chk <= 0)
        return -1;

    return chk;
}

int abcdk_rtsp_server_play_stream(abcdk_rtsp_server_t *ctx, int media, int stream, const void *data, size_t size, int64_t dur)
{
    int chk;

    assert(ctx != NULL && media > 0 && stream > 0 && data != NULL && size > 0 && dur >= 0);

    ABCDK_ASSERT(!ctx->worker_flag, TT("服务尚未启动，禁止修改运行配置。"));

    chk = ctx->l5_server_ctx->play_stream(media, stream, data, size, dur);
    if (chk != 0)
        return -1;

    return 0;
}

#else //_RTSP_SERVER_HH

#endif //_RTSP_SERVER_HH

__END_DECLS