/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/serialport.h"

/** 串口通讯对象。*/
struct _abcdk_serialport
{
    /** 句柄。*/
    int fd;

    /** 互斥量。*/
    abcdk_mutex_t mutex;

    /** 间隔(毫秒)。*/
    uint64_t interval;

}; // abcdk_serialport_t;

void abcdk_serialport_destroy(abcdk_serialport_t **ctx)
{
    abcdk_serialport_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_mutex_destroy(&ctx_p->mutex);
    abcdk_closep(&ctx_p->fd);
    abcdk_heap_free(ctx_p);
}

abcdk_serialport_t *abcdk_serialport_create()
{
    abcdk_serialport_t *ctx;

    ctx = abcdk_heap_alloc(sizeof(abcdk_serialport_t));
    if (!ctx)
        return NULL;

    ctx->fd = -1;
    abcdk_mutex_init2(&ctx->mutex, 0);
    ctx->interval = 0;

    return ctx;
}

int abcdk_serialport_attach(abcdk_serialport_t *ctx, int fd)
{
    int old;

    assert(ctx != NULL && fd >= 0);

    /*添加异步标志。*/
    abcdk_fflag_add(fd, O_NONBLOCK);

    abcdk_mutex_lock(&ctx->mutex, 1);

    old = ctx->fd;
    ctx->fd = fd;

    abcdk_mutex_unlock(&ctx->mutex);

    return old;
}

int abcdk_serialport_detach(abcdk_serialport_t *ctx)
{
    int old;

    assert(ctx != NULL);

    abcdk_mutex_lock(&ctx->mutex, 1);

    old = ctx->fd;
    ctx->fd = -1;

    abcdk_mutex_unlock(&ctx->mutex);

    return old;
}

int abcdk_serialport_set_option(abcdk_serialport_t *ctx,int opt,...)
{
    int chk = 0;

    assert(ctx != NULL);

    va_list vaptr;
    va_start(vaptr, opt);

    abcdk_mutex_lock(&ctx->mutex, 1);

    switch (opt)
    {
    case ABCDK_SERIALPORT_OPT_INTERVAL:
        ctx->interval = va_arg(vaptr, uint64_t);
        break;
    default:
        chk = -1;
        break;
    }

    abcdk_mutex_unlock(&ctx->mutex);

    va_end(vaptr);

    return chk;
}

int abcdk_serialport_get_option(abcdk_serialport_t *ctx,int opt,...)
{
    int chk = 0;

    assert(ctx != NULL);

    va_list vaptr;
    va_start(vaptr, opt);

    abcdk_mutex_lock(&ctx->mutex, 1);

    switch (opt)
    {
    case ABCDK_SERIALPORT_OPT_INTERVAL:
        *(va_arg(vaptr, uint64_t*)) = ctx->interval;
        break;
    default:
        chk = -1;
        break;
    }

    abcdk_mutex_unlock(&ctx->mutex);

    va_end(vaptr);

    return chk;
}

int _abcdk_serialport_transfer_nonsafe(abcdk_serialport_t *ctx, const void *out, size_t outlen, void *in, size_t inlen,
                                       time_t timeout, const void *magic, size_t mglen)
{
    ssize_t wlen, rlen;
    int chk;

    /*两组命令之间的间隔(毫秒)。*/
    if (ctx->interval > 0)
        usleep(ctx->interval*1000);

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

int abcdk_serialport_transfer(abcdk_serialport_t *ctx, const void *out, size_t outlen, void *in, size_t inlen,
                              time_t timeout, const void *magic, size_t mglen)
{
    int chk;

    assert(ctx != NULL && timeout > 0);

    abcdk_mutex_lock(&ctx->mutex, 1);

    chk = _abcdk_serialport_transfer_nonsafe(ctx, out, outlen, in, inlen, timeout, magic, mglen);

    abcdk_mutex_unlock(&ctx->mutex);

    return chk;
}