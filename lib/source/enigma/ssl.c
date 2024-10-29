/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/enigma/ssl.h"

/**基于enigma的SSL通讯。 */
struct _abcdk_enigma_ssl
{
    /**发送加密环境。*/
    abcdk_enigma_t *en_send_ctx;

    /**接收加密环境。*/
    abcdk_enigma_t *en_recv_ctx;

    /**发送队列。*/
    abcdk_tree_t *send_queue;

    /**发送游标。*/
    size_t send_pos;

    /**重发指针和长度。*/
    const void *send_repeated_p;
    size_t send_repeated_l;

    /**接收队列。*/
    abcdk_stream_t *recv_queue;

    /**接收缓存。*/
    abcdk_object_t *recv_buf;

    /**盐发送状态。0 未发送，!0 已发送。 */
    int send_sprinkle_salt;

    /**已接收的盐长度。*/
    size_t recv_salt_len;

    /** 发送句柄。*/
    int send_fd;

    /** 接收句柄。*/
    int recv_fd;

}; // abcdk_enigma_ssl_t;

void abcdk_enigma_ssl_destroy(abcdk_enigma_ssl_t **ctx)
{
    abcdk_enigma_ssl_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_enigma_free(&ctx_p->en_recv_ctx);
    abcdk_enigma_free(&ctx_p->en_send_ctx);
    abcdk_tree_free(&ctx_p->send_queue);
    abcdk_stream_destroy(&ctx_p->recv_queue);
    abcdk_object_unref(&ctx_p->recv_buf);

    abcdk_heap_free(ctx_p);
}

int _abcdk_enigma_ssl_init(abcdk_enigma_ssl_t *ctx, const uint8_t *key, size_t size)
{
    uint8_t hashcode[32];
    uint64_t send_seed[4] = {0}, recv_seed[4] = {0};
    int chk;

    /*密钥转换为定长HASHCODE。*/
    chk = abcdk_sha256_once(key, size, hashcode);
    if (chk != 0)
        return -1;

    /*分解成4个64位整数。不能直接复制内存，因为存在大小端存储顺序不同的问题。*/
    for (int i = 0; i < 32; i++)
    {
        send_seed[i % 4] <<= 8;
        send_seed[i % 4] |= (uint64_t)hashcode[i];
        recv_seed[i % 4] <<= 8;
        recv_seed[i % 4] |= (uint64_t)hashcode[i];
    }

    ctx->en_send_ctx = abcdk_enigma_create3(send_seed, 4,256);
    ctx->en_recv_ctx = abcdk_enigma_create3(recv_seed, 4,256);

    if (!ctx->en_send_ctx || !ctx->en_recv_ctx)
        return -2;

    ctx->send_queue = abcdk_tree_alloc3(1);
    if (!ctx->send_queue)
        return -3;

    ctx->recv_queue = abcdk_stream_create();
    if (!ctx->recv_queue)
        return -4;

    ctx->recv_buf = abcdk_object_alloc2(64*1024);
    if (!ctx->recv_buf)
        return -5;

    ctx->send_fd = -1;
    ctx->recv_fd = -1;
    ctx->send_pos = 0;
    ctx->send_repeated_p = NULL;
    ctx->send_repeated_l = 0;
    ctx->send_sprinkle_salt = 0;
    ctx->recv_salt_len = 0;

    return 0;
}

abcdk_enigma_ssl_t *abcdk_enigma_ssl_create(const uint8_t *key, size_t size)
{
    abcdk_enigma_ssl_t *ctx;
    int chk;

    assert(key != NULL && size > 0);

    ctx = (abcdk_enigma_ssl_t *)abcdk_heap_alloc(sizeof(abcdk_enigma_ssl_t));
    if (!ctx)
        return NULL;

    chk = _abcdk_enigma_ssl_init(ctx, key, size);
    if (chk != 0)
        goto ERR;

    return ctx;

ERR:

    abcdk_enigma_ssl_destroy(&ctx);
    return NULL;
}

abcdk_enigma_ssl_t *abcdk_enigma_ssl_create_from_file(const char *file)
{
    abcdk_enigma_ssl_t *ctx;
    abcdk_object_t *key;

    assert(file != NULL);

    key = abcdk_mmap_filename(file, 0, 0, 0, 0);
    if (!key)
        return NULL;

    ctx = abcdk_enigma_ssl_create(key->pptrs[0], key->sizes[0]);
    abcdk_object_unref(&key);
    if (!ctx)
        return NULL;

    return ctx;
}

int abcdk_enigma_ssl_set_fd(abcdk_enigma_ssl_t *ctx, int fd, int flag)
{
    assert(ctx != NULL && fd >= 0);

    if (flag == 0)
    {
        ctx->send_fd = ctx->recv_fd = fd;
    }
    else if (flag == 1)
    {
        ctx->recv_fd = fd;
    }
    else if (flag == 2)
    {
        ctx->recv_fd = fd;
    }
    else
    {
        return -1;
    }

    return 0;
}

int abcdk_enigma_ssl_get_fd(abcdk_enigma_ssl_t *ctx, int flag)
{
    int old;

    assert(ctx != NULL);

    if (flag == 0)
    {
        if (ctx->recv_fd == ctx->send_fd)
            return ctx->send_fd;
        else
            return -1;
    }
    else if (flag == 1)
    {
        return ctx->recv_fd;
    }
    else if (flag == 2)
    {
        return ctx->send_fd;
    }

    return -1;
}

ssize_t _abcdk_enigma_ssl_write(abcdk_enigma_ssl_t *ctx, const void *data, size_t size)
{
    uint64_t salt_seed = 0;
    char salt[256 + 1] = {0};
    abcdk_tree_t *en_data = NULL;
    abcdk_tree_t *p = NULL;
    ssize_t slen = 0;

    assert(ctx != NULL && data != NULL && size > 0);

    /*发送前先撒盐。*/
    if (!ctx->send_sprinkle_salt)
    {
        en_data = abcdk_tree_alloc3(256);
        if (!en_data)
            return 0; // 内存不足时，关闭当前句柄。

        /*使用单字节的所有字符生成盐。*/
        for(int i = 0;i<256;i++)
            salt[i] = i;

        /*使用洗牌算法把盐搅拌一下。*/
        salt_seed = abcdk_rand_q();
        abcdk_rand_shuffle_array(salt,256,&salt_seed,1);

        /*加密。*/
        abcdk_enigma_light_batch(ctx->en_send_ctx, en_data->obj->pptrs[0], salt, 256);

        /*追加到发送队列末尾。*/
        abcdk_tree_insert2(ctx->send_queue, en_data, 0);

        /*撒盐一次即可。*/
        ctx->send_sprinkle_salt = 1;
    }

    /*警告：如果参数的指针和长度未改变，则认为是管道空闲重发。由于前一次调用已经对数据进行加密并加入待发送对列，因此忽略即可。*/
    if (ctx->send_repeated_p != data || ctx->send_repeated_l != size)
    {
        en_data = abcdk_tree_alloc3(size);
        if (!en_data)
            return 0; // 内存不足时，关闭当前句柄。

        /*记录指针和长度，重发时会检测这两个值。*/
        ctx->send_repeated_p = data;
        ctx->send_repeated_l = size;

        /*加密。*/
        abcdk_enigma_light_batch(ctx->en_send_ctx, en_data->obj->pptrs[0], data, size);

        /*追加到发送队列末尾。*/
        abcdk_tree_insert2(ctx->send_queue, en_data, 0);
    }

NEXT_MSG:

    p = abcdk_tree_child(ctx->send_queue, 1);

    /*通知应用层，发送队列空闲。*/
    if (!p)
    {
        ctx->send_repeated_p = NULL;
        ctx->send_repeated_l = 0;
        return size;
    }

    assert(ctx->send_fd >= 0);

    /*
     * 发。
     *
     * 警告：补发数据时参数不能改变(指针和长度)。
     */
    slen = write(ctx->send_fd, ABCDK_PTR2VPTR(p->obj->pptrs[0], ctx->send_pos), p->obj->sizes[0] - ctx->send_pos);
    if (slen < 0)
        return -1;
    else if (slen == 0)
        return 0;

    /*滚动发送游标。*/
    ctx->send_pos += slen;

    /*当前节点未发送完整，则继续发送。*/
    if (ctx->send_pos < p->obj->sizes[0])
        goto NEXT_MSG;

    /*发送游标归零。*/
    ctx->send_pos = 0;

    /*从队列中删除已经发送完整的节点。*/
    abcdk_tree_unlink(p);
    abcdk_tree_free(&p);

    /*并继续发送剩余节点。*/
    goto NEXT_MSG;
}

ssize_t abcdk_enigma_ssl_write(abcdk_enigma_ssl_t *ctx, const void *data, size_t size)
{
    ssize_t slen = 0, alen = 0;

    assert(ctx != NULL && data != NULL && size > 0);

    while (alen < size)
    {
        /*分块发送。*/
        slen = _abcdk_enigma_ssl_write(ctx, ABCDK_PTR2VPTR(data, alen), ABCDK_MIN(size - alen, (size_t)(64*1024)));
        if (slen < 0)
            return (alen > 0 ? alen : -1); // 优先返回已发送的数据长度。
        else if (slen == 0)
            return (alen > 0 ? alen : 0); // 优先返回已发送的数据长度。

        alen += slen;
    }

    return alen;
}

ssize_t abcdk_enigma_ssl_read(abcdk_enigma_ssl_t *ctx, void *data, size_t size)
{
    char salt[256 + 1] = {0};
    abcdk_object_t *de_data = NULL;
    ssize_t rlen = 0, alen = 0;
    int chk;

    assert(ctx != NULL && data != NULL && size > 0);

NEXT_LOOP:

    /*如果数据存在盐则先读取盐。*/
    if (ctx->recv_salt_len < 256)
    {
        rlen = abcdk_stream_read(ctx->recv_queue, ABCDK_PTR2VPTR(salt, ctx->recv_salt_len), 256 - ctx->recv_salt_len);
        if (rlen > 0)
            ctx->recv_salt_len += rlen;
    }

    /*盐读取完成后，才是真实数据。*/
    if (256 == ctx->recv_salt_len)
    {
        rlen = abcdk_stream_read(ctx->recv_queue, ABCDK_PTR2VPTR(data, alen), size - alen);
        if (rlen > 0)
            alen += rlen;

        if (alen >= size)
            return alen;
    }

    assert(ctx->recv_fd >= 0);

    /*收。*/
    rlen = read(ctx->recv_fd, ctx->recv_buf->pptrs[0], ctx->recv_buf->sizes[0]);
    if (rlen < 0)
        return (alen > 0 ? alen : -1); // 优先返回已接收的数据长度。
    else if (rlen == 0)
        return (alen > 0 ? alen : 0); // 优先返回已接收的数据长度。

    de_data = abcdk_object_alloc2(rlen);
    if (!de_data)
        return 0; // 内存不足时，关闭当前句柄。

    /*解密。*/
    abcdk_enigma_light_batch(ctx->en_recv_ctx, de_data->pptrs[0], ctx->recv_buf->pptrs[0], rlen);

    /*追加到接收队列。*/
    chk = abcdk_stream_write(ctx->recv_queue, de_data);
    if (chk != 0)
    {
        abcdk_object_unref(&de_data);
        return 0; // 内存不足时，关闭当前句柄。
    }

    goto NEXT_LOOP;
}
