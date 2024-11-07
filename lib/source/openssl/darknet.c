/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/openssl/darknet.h"

#ifdef OPENSSL_VERSION_NUMBER

/**简单的安全套接字。*/
struct _abcdk_openssl_darknet
{
    /**方案。*/
    int scheme;

    /*密钥。*/
    uint8_t key[32];

    /*向量。*/
    uint8_t send_iv[256];
    uint8_t recv_iv[256];

    /**向量的长度。*/
    size_t iv_len;


    /**AES环境。*/
    EVP_CIPHER_CTX *aes_send_ctx;
    EVP_CIPHER_CTX *aes_recv_ctx;


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

    /**接收数据包。*/
    abcdk_receiver_t *recv_pack;

    /**向量是否已经发送。*/
    int send_iv_ok;

    /**已接收的向量的长度。*/
    size_t recv_iv_len;

    /** 发送句柄。*/
    int send_fd;

    /** 接收句柄。*/
    int recv_fd;

}; // abcdk_openssl_darknet_t;

void abcdk_openssl_darknet_destroy(abcdk_openssl_darknet_t **ctx)
{
    abcdk_openssl_darknet_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if(ctx_p->aes_recv_ctx)
    {
        EVP_CIPHER_CTX_cleanup(ctx_p->aes_recv_ctx);
        EVP_CIPHER_CTX_free(ctx_p->aes_recv_ctx);
    }

    if(ctx_p->aes_send_ctx)
    {
        EVP_CIPHER_CTX_cleanup(ctx_p->aes_send_ctx);
        EVP_CIPHER_CTX_free(ctx_p->aes_send_ctx);
    }


    abcdk_tree_free(&ctx_p->send_queue);
    abcdk_stream_destroy(&ctx_p->recv_queue);
    abcdk_object_unref(&ctx_p->recv_buf);
    abcdk_receiver_unref(&ctx_p->recv_pack);

    abcdk_heap_free(ctx_p);
}

static int _abcdk_openssl_darknet_init(abcdk_openssl_darknet_t *ctx, const uint8_t *key, size_t size)
{
    int chk;

    /*生成密钥。*/
    abcdk_sha256_once(key,size,ctx->key);

    if(ctx->scheme == ABCDK_OPENSSL_DARKNET_SCHEME_AES256CTR)
    {
        ctx->aes_send_ctx = EVP_CIPHER_CTX_new();
        ctx->aes_recv_ctx = EVP_CIPHER_CTX_new();

        if (!ctx->aes_send_ctx || !ctx->aes_recv_ctx)
            return -2;

        chk = EVP_CipherInit_ex(ctx->aes_send_ctx, EVP_aes_256_ctr(), NULL, NULL, NULL, 1);
        chk = EVP_CipherInit_ex(ctx->aes_recv_ctx, EVP_aes_256_ctr(), NULL, NULL, NULL, 0);

        ctx->iv_len = 16;
    }
    else 
    {
        return -22;
    }

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
    ctx->send_iv_ok = 0;
    ctx->recv_iv_len = 0;

    return 0;
}


abcdk_openssl_darknet_t *abcdk_openssl_darknet_create(int scheme, const uint8_t *key, size_t size)
{
    abcdk_openssl_darknet_t *ctx;
    int chk;

    assert(key != NULL && size > 0);

    ctx = (abcdk_openssl_darknet_t *)abcdk_heap_alloc(sizeof(abcdk_openssl_darknet_t));
    if (!ctx)
        return NULL;

    ctx->scheme = scheme;
    chk = _abcdk_openssl_darknet_init(ctx, key, size);
    if (chk != 0)
        goto ERR;
    
    return ctx;

ERR:

    abcdk_openssl_darknet_destroy(&ctx);
    return NULL;
}

abcdk_openssl_darknet_t *abcdk_openssl_darknet_create_from_file(int scheme, const char *file)
{
    abcdk_openssl_darknet_t *ctx;
    abcdk_object_t *key;

    assert(file != NULL);

    key = abcdk_mmap_filename(file, 0, 0, 0, 0);
    if (!key)
        return NULL;

    ctx = abcdk_openssl_darknet_create(scheme,key->pptrs[0], key->sizes[0]);
    abcdk_object_unref(&key);
    if (!ctx)
        return NULL;

    return ctx;
}

int abcdk_openssl_darknet_set_fd(abcdk_openssl_darknet_t *ctx, int fd, int flag)
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

int abcdk_openssl_darknet_get_fd(abcdk_openssl_darknet_t *ctx, int flag)
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

static int _abcdk_openssl_darknet_write_init(abcdk_openssl_darknet_t *ctx)
{
    int chk;

    if (ctx->scheme == ABCDK_OPENSSL_DARKNET_SCHEME_AES256CTR)
    {
        if (EVP_CipherInit_ex(ctx->aes_send_ctx, NULL, NULL, ctx->key, ctx->send_iv, 1) != 1)
            return -1;
    }
    else
    {
        return -1;
    }

    return 0;
}

static abcdk_tree_t *_abcdk_openssl_darknet_write_update(abcdk_openssl_darknet_t *ctx, const void *in, int in_len)
{
    abcdk_tree_t *en_data = NULL;
    int en_outlen;

    en_data = abcdk_tree_alloc3(in_len);
    if (!en_data)
        return NULL; 

    if(ctx->scheme == ABCDK_OPENSSL_DARKNET_SCHEME_AES256CTR)
    {
        EVP_CipherUpdate(ctx->aes_send_ctx, en_data->obj->pptrs[0], &en_outlen, in, in_len);
        assert(en_outlen == in_len);
    }
    else
    {
        abcdk_tree_free(&en_data);
    }

    return en_data;
}

static ssize_t _abcdk_openssl_darknet_write_fragment(abcdk_openssl_darknet_t *ctx, const void *data, size_t size)
{
    char salt[256 + 1] = {0};
    abcdk_tree_t *en_data = NULL;
    abcdk_tree_t *p = NULL;
    ssize_t slen = 0;
    int chk;

    /*先发送向量。*/
    if (!ctx->send_iv_ok)
    {
        /*生成向量。*/
#ifdef OPENSSL_VERSION_NUMBER
        RAND_bytes(ctx->send_iv,ctx->iv_len);
#else 
        abcdk_rand_bytes(ctx->send_iv,ctx->iv_len,5);
#endif //OPENSSL_VERSION_NUMBER

        /*初始化加密算法环境。*/
        chk = _abcdk_openssl_darknet_write_init(ctx);
        if (chk != 0)
            return 0; // 内存不足时，关闭当前句柄。

        en_data = abcdk_tree_alloc4(ctx->send_iv,ctx->iv_len);
        if (!en_data)
            return 0; // 内存不足时，关闭当前句柄。

        /*追加到发送队列末尾。*/
        abcdk_tree_insert2(ctx->send_queue, en_data, 0);

        /*发送一次即可。*/
        ctx->send_iv_ok = 1;
    }

    /*警告：如果参数的指针和长度未改变，则认为是管道空闲重发。由于前一次调用已经对数据进行加密并加入待发送对列，因此忽略即可。*/
    if (ctx->send_repeated_p != data || ctx->send_repeated_l != size)
    {
        /*记录指针和长度，重发时会检测这两个值。*/
        ctx->send_repeated_p = data;
        ctx->send_repeated_l = size;

        /*加密。*/
        en_data = _abcdk_openssl_darknet_write_update(ctx, data, size);
        if (!en_data)
            return 0; // 内存不足时，关闭当前句柄。

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
        return size;//在这里返回发送数据的实际长度。
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

ssize_t abcdk_openssl_darknet_write(abcdk_openssl_darknet_t *ctx, const void *data, size_t size)
{
    ssize_t slen = 0, alen = 0;

    assert(ctx != NULL && data != NULL && size > 0);

    while (alen < size)
    {
        /*分块发送。*/
        slen = _abcdk_openssl_darknet_write_fragment(ctx, ABCDK_PTR2VPTR(data, alen), ABCDK_MIN(size - alen, (size_t)(65535)));
        if (slen < 0)
            return (alen > 0 ? alen : -1); // 优先返回已发送的数据长度。
        else if (slen == 0)
            return (alen > 0 ? alen : 0); // 优先返回已发送的数据长度。

        alen += slen;
    }

    return alen;
}

static int _abcdk_openssl_darknet_read_init(abcdk_openssl_darknet_t *ctx)
{
    int chk;

    if (ctx->scheme == ABCDK_OPENSSL_DARKNET_SCHEME_AES256CTR)
    {
        if (EVP_CipherInit_ex(ctx->aes_recv_ctx, NULL, NULL, ctx->key, ctx->recv_iv, 0) != 1)
            return -1;
    }
    else
    {
        return -1;
    }

    return 0;
}

static abcdk_object_t *_abcdk_openssl_darknet_read_update(abcdk_openssl_darknet_t *ctx, const void *in, int in_len)
{
    abcdk_object_t *de_data = NULL;
    int de_outlen;

    de_data = abcdk_object_alloc2(in_len);
    if (!de_data)
        return NULL; 

    if(ctx->scheme == ABCDK_OPENSSL_DARKNET_SCHEME_AES256CTR)
    {
        EVP_CipherUpdate(ctx->aes_recv_ctx, de_data->pptrs[0], &de_outlen, in, in_len);
        assert(de_outlen == in_len);
    }
    else
    {
        abcdk_object_unref(&de_data);
    }

    return de_data;
}


ssize_t abcdk_openssl_darknet_read(abcdk_openssl_darknet_t *ctx, void *data, size_t size)
{
    abcdk_object_t *de_data = NULL;
    ssize_t rlen = 0;
    int chk;

    assert(ctx != NULL && data != NULL && size > 0);

NEXT_LOOP:

    if (ctx->iv_len == ctx->recv_iv_len)
    {
        rlen = abcdk_stream_read(ctx->recv_queue, data, size);
        if (rlen > 0)
            return rlen;
    }

    assert(ctx->recv_fd >= 0);

    /*收。*/
    rlen = read(ctx->recv_fd, ctx->recv_buf->pptrs[0], ctx->recv_buf->sizes[0]);
    if (rlen < 0)
        return -1;
    else if (rlen == 0)
        return 0;

    /*先读取向量。*/
    if (ctx->recv_iv_len < ctx->iv_len)
    {
        /*计算待提取的向量长度并提取。*/
        size_t diff = ABCDK_MIN(ctx->iv_len - ctx->recv_iv_len, rlen);
        memcpy(ctx->recv_iv + ctx->recv_iv_len, ctx->recv_buf->pptrs[0], diff);
        ctx->recv_iv_len += diff;

        /*当向量读取完整后，进行初始化。*/
        if (ctx->recv_iv_len == ctx->iv_len)
        {
            chk = _abcdk_openssl_darknet_read_init(ctx);
            if (chk != 0)
                return 0;
        }

        /*如果缓存没有剩余数据，则直接返回。*/
        if (rlen - diff <= 0)
            return -1;

        /*处理剩余数据，解密。*/
        de_data = _abcdk_openssl_darknet_read_update(ctx,ctx->recv_buf->pptrs[0] + diff, rlen - diff);
        if (!de_data)
            return 0; // 内存不足时，关闭当前句柄。
    }
    else
    {
        /*解密。*/
        de_data = _abcdk_openssl_darknet_read_update(ctx,ctx->recv_buf->pptrs[0], rlen);
        if (!de_data)
            return 0; // 内存不足时，关闭当前句柄。
    }

    /*追加到接收队列。*/
    chk = abcdk_stream_write(ctx->recv_queue, de_data);
    if (chk != 0)
    {
        abcdk_object_unref(&de_data);
        return 0; // 内存不足时，关闭当前句柄。
    }

    goto NEXT_LOOP;
}

#endif //OPENSSL_VERSION_NUMBER