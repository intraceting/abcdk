/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/ssl/ssl.h"

/** 简单的安全套接字。*/
struct _abcdk_ssl
{
    /**方案。*/
    int scheme;

    /**Enigma发送加密环境。*/
    abcdk_enigma_t *enigma_send_ctx;

    /**Enigma接收加密环境。*/
    abcdk_enigma_t *enigma_recv_ctx;

    /**AES发送加密环境。*/
    abcdk_openssl_cipher_t *aes_send_ctx;

    /**AES接收加密环境。*/
    abcdk_openssl_cipher_t *aes_recv_ctx;

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

}; // abcdk_ssl_t;

void abcdk_ssl_destroy(abcdk_ssl_t **ctx)
{
    abcdk_ssl_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_enigma_free(&ctx_p->enigma_recv_ctx);
    abcdk_enigma_free(&ctx_p->enigma_send_ctx);
    abcdk_openssl_cipher_destroy(&ctx_p->aes_recv_ctx);
    abcdk_openssl_cipher_destroy(&ctx_p->aes_send_ctx);
    abcdk_tree_free(&ctx_p->send_queue);
    abcdk_stream_destroy(&ctx_p->recv_queue);
    abcdk_object_unref(&ctx_p->recv_buf);

    abcdk_heap_free(ctx_p);
}

static int _abcdk_ssl_enigma_init(abcdk_ssl_t *ctx, const uint8_t *key, size_t size)
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

    ctx->enigma_send_ctx = abcdk_enigma_create3(send_seed, 4,256);
    ctx->enigma_recv_ctx = abcdk_enigma_create3(recv_seed, 4,256);

    if (!ctx->enigma_send_ctx || !ctx->enigma_recv_ctx)
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

static int _abcdk_ssl_aes_init(abcdk_ssl_t *ctx, const uint8_t *key, size_t size)
{
    if (ctx->scheme == ABCDK_SSL_SCHEME_AES_256_GCM)
    {
        ctx->aes_send_ctx = abcdk_openssl_cipher_create(ABCDK_OPENSSL_CIPHER_SCHEME_AES_256_GCM,key,size);
        ctx->aes_recv_ctx = abcdk_openssl_cipher_create(ABCDK_OPENSSL_CIPHER_SCHEME_AES_256_GCM,key,size);
    }
    else if(ctx->scheme == ABCDK_SSL_SCHEME_AES_256_CBC)
    {
        ctx->aes_send_ctx = abcdk_openssl_cipher_create(ABCDK_OPENSSL_CIPHER_SCHEME_AES_256_CBC,key,size);
        ctx->aes_recv_ctx = abcdk_openssl_cipher_create(ABCDK_OPENSSL_CIPHER_SCHEME_AES_256_CBC,key,size);
    }

    if (!ctx->aes_send_ctx || !ctx->aes_recv_ctx)
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

abcdk_ssl_t *abcdk_ssl_create(int scheme, const uint8_t *key, size_t size)
{
    abcdk_ssl_t *ctx;
    int chk;

    assert(key != NULL && size > 0);

    ctx = (abcdk_ssl_t *)abcdk_heap_alloc(sizeof(abcdk_ssl_t));
    if (!ctx)
        return NULL;

    ctx->scheme = -1;

    if (scheme == ABCDK_SSL_SCHEME_ENIGMA)
    {
        ctx->scheme = scheme;
        chk = _abcdk_ssl_enigma_init(ctx, key, size);
        if (chk != 0)
            goto ERR;
    }
    else if (scheme == ABCDK_SSL_SCHEME_AES_256_GCM || scheme == ABCDK_SSL_SCHEME_AES_256_CBC)
    {
        ctx->scheme = scheme;
        chk = _abcdk_ssl_aes_init(ctx, key, size);
        if (chk != 0)
            goto ERR;
    }

    return ctx;

ERR:

    abcdk_ssl_destroy(&ctx);
    return NULL;
}

abcdk_ssl_t *abcdk_ssl_create_from_file(int scheme, const char *file)
{
    abcdk_ssl_t *ctx;
    abcdk_object_t *key;

    assert(file != NULL);

    key = abcdk_mmap_filename(file, 0, 0, 0, 0);
    if (!key)
        return NULL;

    ctx = abcdk_ssl_create(scheme,key->pptrs[0], key->sizes[0]);
    abcdk_object_unref(&key);
    if (!ctx)
        return NULL;

    return ctx;
}

int abcdk_ssl_set_fd(abcdk_ssl_t *ctx, int fd, int flag)
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

int abcdk_ssl_get_fd(abcdk_ssl_t *ctx, int flag)
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

static ssize_t _abcdk_ssl_enigma_write_fragment(abcdk_ssl_t *ctx, const void *data, size_t size)
{
    uint64_t salt_seed = 0;
    char salt[256 + 1] = {0};
    abcdk_tree_t *en_data = NULL;
    abcdk_tree_t *p = NULL;
    ssize_t slen = 0;

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
        abcdk_enigma_light_batch(ctx->enigma_send_ctx, en_data->obj->pptrs[0], salt, 256);

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
        abcdk_enigma_light_batch(ctx->enigma_send_ctx, en_data->obj->pptrs[0], data, size);

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

    /*发。*/
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

static ssize_t _abcdk_ssl_enigma_write(abcdk_ssl_t *ctx, const void *data, size_t size)
{
    ssize_t slen = 0, alen = 0;

    while (alen < size)
    {
        /*分块发送。*/
        slen = _abcdk_ssl_enigma_write_fragment(ctx, ABCDK_PTR2VPTR(data, alen), ABCDK_MIN(size - alen, (size_t)(64*1024)));
        if (slen < 0)
            return (alen > 0 ? alen : -1); // 优先返回已发送的数据长度。
        else if (slen == 0)
            return (alen > 0 ? alen : 0); // 优先返回已发送的数据长度。

        alen += slen;
    }

    return alen;
}

static ssize_t _abcdk_ssl_enigma_read(abcdk_ssl_t *ctx, void *data, size_t size)
{
    char salt[256 + 1] = {0};
    abcdk_object_t *de_data = NULL;
    ssize_t rlen = 0, alen = 0;
    int chk;

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
    abcdk_enigma_light_batch(ctx->enigma_recv_ctx, de_data->pptrs[0], ctx->recv_buf->pptrs[0], rlen);

    /*追加到接收队列。*/
    chk = abcdk_stream_write(ctx->recv_queue, de_data);
    if (chk != 0)
    {
        abcdk_object_unref(&de_data);
        return 0; // 内存不足时，关闭当前句柄。
    }

    goto NEXT_LOOP;
}

static ssize_t _abcdk_ssl_aes_write(abcdk_ssl_t *ctx, const void *data, size_t size)
{

}

static ssize_t _abcdk_ssl_aes_read(abcdk_ssl_t *ctx, void *data, size_t size)
{

}

ssize_t abcdk_ssl_write(abcdk_ssl_t *ctx,const void *data,size_t size)
{
    ssize_t chk = 0;

    assert(ctx != NULL && data != NULL && size > 0);

    if(ctx->scheme == ABCDK_SSL_SCHEME_ENIGMA)
    {
        return _abcdk_ssl_enigma_write(ctx,data,size);
    }
    else if (ctx->scheme == ABCDK_SSL_SCHEME_AES_256_GCM || ctx->scheme == ABCDK_SSL_SCHEME_AES_256_CBC)
    {
        return _abcdk_ssl_aes_write(ctx,data,size);
    }

    return -1;
}

ssize_t abcdk_ssl_read(abcdk_ssl_t *ctx,void *data,size_t size)
{
    ssize_t chk = 0;

    assert(ctx != NULL && data != NULL && size > 0);
    
    if(ctx->scheme == ABCDK_SSL_SCHEME_ENIGMA)
    {
        return _abcdk_ssl_enigma_read(ctx,data,size);
    }
    else if (ctx->scheme == ABCDK_SSL_SCHEME_AES_256_GCM || ctx->scheme == ABCDK_SSL_SCHEME_AES_256_CBC)
    {
        return _abcdk_ssl_aes_read(ctx,data,size);
    }

    return -1;
}