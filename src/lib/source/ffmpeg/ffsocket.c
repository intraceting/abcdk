/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/ffmpeg/ffsocket.h"

/*简单的IO通讯。*/
struct _abcdk_ffsocket
{
    /**魔法数。 */
    uint32_t magic;
#define ABCDK_FFSOCKET_MAGIC 123456789

    /**SOCKET地址。*/
    abcdk_sockaddr_t sock_addr;

    /**SOCKET监听句柄。*/
    int sock_listen_fd;

    /**SOCKET通讯句柄。*/
    int sock_connect_fd;

    /**地址。 */
    char *addr;

    /**超时(秒)。*/
    int timeout;

    /**证书。 */
    char *cert;

    /**私钥。 */
    char *key;

    /**CA证书路径。 */
    char *capath;

}; // abcdk_ffsocket_t;

void abcdk_ffsocket_destroy(abcdk_ffsocket_t **ctx)
{
    abcdk_ffsocket_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_closep(&ctx_p->sock_listen_fd);
    abcdk_closep(&ctx_p->sock_connect_fd);

    abcdk_heap_free(ctx_p->addr);
    abcdk_heap_free(ctx_p->cert);
    abcdk_heap_free(ctx_p->key);
    abcdk_heap_free(ctx_p->capath);

    abcdk_heap_free(ctx_p);
}

abcdk_ffsocket_t *abcdk_ffsocket_create(const char *addr, int timeout, const char *cert, const char *key, const char *capath)
{
    abcdk_ffsocket_t *ctx;

    assert(addr != NULL);

    ctx = (abcdk_ffsocket_t *)abcdk_heap_alloc(sizeof(abcdk_ffsocket_t));
    if (!ctx)
        return NULL;

    ctx->magic = ABCDK_FFSOCKET_MAGIC;
    ctx->sock_connect_fd = -1;
    ctx->sock_listen_fd = -1;

    ctx->addr = (addr ? abcdk_heap_clone(addr, strlen(addr)) : NULL);
    ctx->timeout = ABCDK_CLAMP(timeout, 1, 180);
    ctx->cert = (cert ? abcdk_heap_clone(cert, strlen(cert)) : NULL);
    ctx->key = (key ? abcdk_heap_clone(key, strlen(key)) : NULL);
    ctx->capath = (capath ? abcdk_heap_clone(capath, strlen(capath)) : NULL);

    return ctx;
}

int abcdk_ffsocket_read(void *opaque, uint8_t *buf, int size)
{
    abcdk_ffsocket_t *ctx = (abcdk_ffsocket_t *)opaque;
    int chk;

    assert(ctx->magic == ABCDK_FFSOCKET_MAGIC);

    /*先建立监听。*/
    if (ctx->sock_listen_fd < 0)
    {
        chk = abcdk_sockaddr_from_string(&ctx->sock_addr, ctx->addr, 1);
        if (chk != 0)
            return AVERROR_EOF;

        ctx->sock_listen_fd = abcdk_socket(ctx->sock_addr.family, 0);
        if (ctx->sock_listen_fd < 0)
            return AVERROR_EOF;

        if (ctx->sock_addr.family == AF_UNIX)
        {
            unlink(ctx->sock_addr.addr_un.sun_path);//删除旧的。
            abcdk_mkdir(ctx->sock_addr.addr_un.sun_path,0755);//创建不存的路径。
        }

        chk = abcdk_bind(ctx->sock_listen_fd, &ctx->sock_addr);
        if (chk != 0)
            return AVERROR_EOF;
        
        if (ctx->sock_addr.family == AF_UNIX)
            chmod(ctx->sock_addr.addr_un.sun_path, 0666); // 所有用户都可以连接。

        chk = listen(ctx->sock_listen_fd, 1);
        if (chk != 0)
            return AVERROR_EOF;


        /*改为异步的。*/
        abcdk_fflag_add(ctx->sock_listen_fd, O_NONBLOCK);
    }

    /*等待连接。*/
    if (ctx->sock_connect_fd < 0)
    {
        chk = abcdk_poll(ctx->sock_listen_fd, 0x01, ctx->timeout * 1000);
        if (chk <= 0)
        {
            abcdk_closep(&ctx->sock_listen_fd);
            return AVERROR_EOF;
        }

        ctx->sock_connect_fd = abcdk_accept(ctx->sock_listen_fd, NULL);
        if (ctx->sock_connect_fd < 0)
            return AVERROR(EAGAIN);

        /*改为异步的。*/
        abcdk_fflag_add(ctx->sock_connect_fd, O_NONBLOCK);
    }

    /*等待数据到达。*/
    chk = abcdk_poll(ctx->sock_connect_fd, 0x01, ctx->timeout * 1000);
    if (chk <= 0)
    {
        abcdk_closep(&ctx->sock_connect_fd);
        return AVERROR_EOF;
    }

    /*异步句柄，返回值要分别处理。*/
    chk = read(ctx->sock_connect_fd, buf, size);
    if (chk == 0) // 连接已断开。
    {
        abcdk_closep(&ctx->sock_connect_fd);
        return AVERROR_EOF;
    }
    else if (chk < 0) // 没有更多数据。
    {
        return AVERROR(EAGAIN);
    }

    return chk;
}

int abcdk_ffsocket_write(void *opaque, uint8_t *buf, int size)
{
    abcdk_ffsocket_t *ctx = (abcdk_ffsocket_t *)opaque;
    int chk;

    assert(ctx->magic == ABCDK_FFSOCKET_MAGIC);

    /*先建立连接。*/
    if (ctx->sock_connect_fd < 0)
    {
        chk = abcdk_sockaddr_from_string(&ctx->sock_addr, ctx->addr, 1);
        if (chk != 0)
            return AVERROR_EOF;

        ctx->sock_connect_fd = abcdk_socket(ctx->sock_addr.family, 0);
        if (ctx->sock_connect_fd < 0)
            return AVERROR_EOF;

        /*改为异步的。*/
        abcdk_fflag_add(ctx->sock_connect_fd, O_NONBLOCK);

        chk = abcdk_connect(ctx->sock_connect_fd, &ctx->sock_addr);
        if (chk != 0)
            return AVERROR_EOF;

        /*等待连接完成。*/
        chk = abcdk_poll(ctx->sock_connect_fd, 0x02, ctx->timeout * 1000);
        if (chk <= 0)
            return AVERROR_EOF;
    }

    /*等待链路空闲。不能死等，不然无法在随机位置关闭。*/
    chk = abcdk_poll(ctx->sock_connect_fd, 0x02, 1000);
    if (chk < 0)
    {
        abcdk_closep(&ctx->sock_connect_fd);
        return AVERROR_EOF;
    }
    else if (chk == 0)
    {
        AVERROR(EAGAIN);
    }

    /*异步句柄，返回值要分别处理。*/
    chk = write(ctx->sock_connect_fd,buf,size);
    if (chk == 0) // 连接已断开。
    {
        abcdk_closep(&ctx->sock_connect_fd);
        return AVERROR_EOF;
    }
    else if (chk < 0) // 链路忙。
    {
        return AVERROR(EAGAIN);
    }

    return chk;
}