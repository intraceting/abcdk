/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/stream.h"

/** 流对象。*/
struct _abcdk_stream
{
    /** 句柄。*/
    int fd;

    /** 互斥量。*/
    abcdk_mutex_t mutex;

    /** 间隔(毫秒)。*/
    uint64_t interval;

}; // abcdk_stream_t;

void abcdk_stream_destroy(abcdk_stream_t **ctx)
{
    abcdk_stream_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_mutex_destroy(&ctx_p->mutex);
    abcdk_closep(&ctx_p->fd);
    abcdk_heap_free(ctx_p);
}

abcdk_stream_t *abcdk_stream_create()
{
    abcdk_stream_t *ctx;

    ctx = abcdk_heap_alloc(sizeof(abcdk_stream_t));
    if (!ctx)
        return NULL;

    ctx->fd = -1;
    abcdk_mutex_init2(&ctx->mutex, 0);
    ctx->interval = 0;

    return ctx;
}

void abcdk_stream_lock(abcdk_stream_t *ctx)
{
    assert(ctx != NULL);

    abcdk_mutex_lock(&ctx->mutex, 1);
}

void abcdk_stream_unlock(abcdk_stream_t *ctx)
{
    assert(ctx != NULL);

    abcdk_mutex_unlock(&ctx->mutex);
}

int abcdk_stream_attach(abcdk_stream_t *ctx, int fd)
{
    int old;

    assert(ctx != NULL && fd >= 0);

    /*添加异步标志。*/
    abcdk_fflag_add(fd, O_NONBLOCK);

    old = ctx->fd;
    ctx->fd = fd;

    return old;
}

int abcdk_stream_detach(abcdk_stream_t *ctx)
{
    int old;

    assert(ctx != NULL);

    old = ctx->fd;
    ctx->fd = -1;

    return old;
}

int abcdk_stream_set_option(abcdk_stream_t *ctx, int opt, ...)
{
    int chk = 0;

    assert(ctx != NULL);

    va_list vaptr;
    va_start(vaptr, opt);

    switch (opt)
    {
    case ABCDK_STREAM_OPT_INTERVAL:
        ctx->interval = va_arg(vaptr, uint64_t);
        break;
    default:
        chk = -1;
        break;
    }

    va_end(vaptr);

    return chk;
}

int abcdk_stream_get_option(abcdk_stream_t *ctx, int opt, ...)
{
    int chk = 0;

    assert(ctx != NULL);

    va_list vaptr;
    va_start(vaptr, opt);

    switch (opt)
    {
    case ABCDK_STREAM_OPT_INTERVAL:
        *(va_arg(vaptr, uint64_t *)) = ctx->interval;
        break;
    default:
        chk = -1;
        break;
    }

    va_end(vaptr);

    return chk;
}

int abcdk_stream_transfer(abcdk_stream_t *ctx, const void *out, size_t outlen, void *in, size_t inlen,
                          time_t timeout, const void *magic, size_t mglen)
{
    ssize_t wlen, rlen;
    int chk;

    assert(ctx != NULL && timeout > 0);

    /*两组命令之间的间隔(毫秒)。*/
    if (ctx->interval > 0)
        usleep(ctx->interval * 1000);

    /*按需发送。*/
    if (out != NULL && outlen > 0)
    {
        wlen = abcdk_transfer(ctx->fd, (void *)out, outlen, 2, timeout, NULL, 0);
        if (wlen != outlen)
            return -1;

        /*等待发送完成。*/
        chk = tcdrain(ctx->fd);
        //       assert(chk == 0);
    }

    /*按需接收。*/
    if (in != NULL && inlen > 0)
    {
        rlen = abcdk_transfer(ctx->fd, in, inlen, 1, timeout, magic, mglen);
        if (rlen != inlen)
            return -1;
    }

    return 0;
}