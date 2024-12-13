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
    /*密钥环境。*/
    RSA *rsa_send_ctx;
    RSA *rsa_recv_ctx;
    EVP_CIPHER_CTX *evp_send_ctx;
    EVP_CIPHER_CTX *evp_recv_ctx;

    /*密钥。*/
    uint8_t evp_send_key[32];
    uint8_t evp_recv_key[32];

    /*向量。*/
    uint8_t evp_send_iv[32];
    uint8_t evp_recv_iv[32];

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

    /**头部缓存。*/
    abcdk_object_t *send_hdr;
    abcdk_object_t *recv_hdr;

    /**头部长度。*/
    size_t hdr_len;

    /**头部是否已经发送。*/
    int send_hdr_ok;

    /**已接收头部的长度。*/
    size_t recv_hdr_len;

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

    abcdk_openssl_rsa_free(&ctx_p->rsa_send_ctx);
    abcdk_openssl_rsa_free(&ctx_p->rsa_recv_ctx);
    abcdk_openssl_evp_cipher_ctx_free(&ctx_p->evp_recv_ctx);
    abcdk_openssl_evp_cipher_ctx_free(&ctx_p->evp_send_ctx);
    abcdk_tree_free(&ctx_p->send_queue);
    abcdk_stream_destroy(&ctx_p->recv_queue);
    abcdk_object_unref(&ctx_p->recv_buf);
    abcdk_receiver_unref(&ctx_p->recv_pack);
    abcdk_object_unref(&ctx_p->send_hdr);
    abcdk_object_unref(&ctx_p->recv_hdr);

    abcdk_heap_free(ctx_p);
}

size_t _abcdk_openssl_darknet_hdr_size(RSA *rsa_ctx,int payload)
{
    abcdk_object_t *src,*dst;
    size_t chk = 0;
    
    src = abcdk_object_alloc2(payload);
    if(!src)
        goto END;

    dst = abcdk_openssl_rsa_update(rsa_ctx,src->pptrs[0],src->sizes[0],1);
    if(!src)
        goto END;

    /*Copy SIZE[0].*/
    chk = dst->sizes[0];

END:

    abcdk_object_unref(&src);
    abcdk_object_unref(&dst);

    return chk;
}

abcdk_openssl_darknet_t *abcdk_openssl_darknet_create(RSA *rsa_ctx, int use_pubkey)
{
    abcdk_openssl_darknet_t *ctx;
    int chk;

    assert(rsa_ctx != NULL);

    ctx = (abcdk_openssl_darknet_t *)abcdk_heap_alloc(sizeof(abcdk_openssl_darknet_t));
    if (!ctx)
        return NULL;

    if(abcdk_openssl_rsa_is_prikey(rsa_ctx) && !use_pubkey)
    {
        ctx->rsa_send_ctx = RSAPrivateKey_dup(rsa_ctx);
        ctx->rsa_recv_ctx = RSAPrivateKey_dup(rsa_ctx);
    }
    else
    {
        ctx->rsa_send_ctx = RSAPublicKey_dup(rsa_ctx);
        ctx->rsa_recv_ctx = RSAPublicKey_dup(rsa_ctx);
    }
    
    if (!ctx->rsa_send_ctx || !ctx->rsa_recv_ctx)
        goto ERR;

    ctx->evp_send_ctx = EVP_CIPHER_CTX_new();
    ctx->evp_recv_ctx = EVP_CIPHER_CTX_new();

    if (!ctx->evp_send_ctx || !ctx->evp_recv_ctx)
        goto ERR;

    ctx->send_queue = abcdk_tree_alloc3(1);
    if (!ctx->send_queue)
        goto ERR;

    ctx->recv_queue = abcdk_stream_create();
    if (!ctx->recv_queue)
        goto ERR;

    ctx->recv_buf = abcdk_object_alloc2(64*1024);
    if (!ctx->recv_buf)
        goto ERR;

    /*计算头部长度。*/
    ctx->hdr_len = _abcdk_openssl_darknet_hdr_size(rsa_ctx, 1 + 32 + 16 + 79);

    ctx->recv_hdr = abcdk_object_alloc2(ctx->hdr_len);
    if (!ctx->recv_hdr)
        goto ERR;

    ctx->send_fd = -1;
    ctx->recv_fd = -1;
    ctx->send_pos = 0;
    ctx->send_repeated_p = NULL;
    ctx->send_repeated_l = 0;
    ctx->send_hdr_ok = 0;
    ctx->recv_hdr_len = 0;


    return ctx;

ERR:

    abcdk_openssl_darknet_destroy(&ctx);
    return NULL;
}

abcdk_openssl_darknet_t *abcdk_openssl_darknet_create_from_file(const char *rsa_file,int pubkey)
{
    abcdk_openssl_darknet_t *ctx;
    RSA *rsa_ctx;

    assert(rsa_file != NULL);

    rsa_ctx = abcdk_openssl_rsa_load(rsa_file, pubkey, NULL);
    if(!rsa_ctx)
        return NULL;

    ctx = abcdk_openssl_darknet_create(rsa_ctx, pubkey);
    abcdk_openssl_rsa_free(&rsa_ctx);

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
        ctx->send_fd = fd;
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
    char hdr[128] = {0};
    int ver = 1;
    int chk;

    /*生成随机密钥和向量。*/
    RAND_bytes(ctx->evp_send_key,32);
    RAND_bytes(ctx->evp_send_iv,16);

    /*填充明文头部。*/
    abcdk_bloom_write_number(hdr, 128, 0, 8, ver);
    memcpy(hdr + 1, ctx->evp_send_key, 32);
    memcpy(hdr + 1 + 32, ctx->evp_send_iv, 16);

    /*用RSA加密头部。*/
    ctx->send_hdr = abcdk_openssl_rsa_update(ctx->rsa_send_ctx, hdr, 128, 1);
    if (!ctx->send_hdr)
        return -1;

    chk = EVP_CipherInit_ex(ctx->evp_send_ctx, EVP_aes_256_ctr(), NULL, ctx->evp_send_key, ctx->evp_send_iv, 1);
    if (chk != 1)
        return -2;

    return 0;
}

static abcdk_tree_t *_abcdk_openssl_darknet_write_update(abcdk_openssl_darknet_t *ctx, const void *in, int in_len)
{
    abcdk_tree_t *en_data = NULL;
    int en_outlen;

    en_data = abcdk_tree_alloc3(in_len);
    if (!en_data)
        return NULL; 

    EVP_CipherUpdate(ctx->evp_send_ctx, en_data->obj->pptrs[0], &en_outlen, in, in_len);
    assert(en_outlen == in_len);

    return en_data;
}

ssize_t abcdk_openssl_darknet_write(abcdk_openssl_darknet_t *ctx, const void *data, size_t size)
{
    char salt[256 + 1] = {0};
    abcdk_tree_t *en_data = NULL;
    abcdk_tree_t *p = NULL;
    ssize_t slen = 0;
    int chk;

    assert(ctx != NULL && data != NULL && size > 0);

    /*先发送头部。*/
    if (!ctx->send_hdr_ok)
    {
        /*初始化发送环境。*/
        chk = _abcdk_openssl_darknet_write_init(ctx);
        if (chk != 0)
            return 0; // 内存不足时，关闭当前句柄。

        /*复制头部。*/
        en_data = abcdk_tree_alloc4(ctx->send_hdr->pptrs[0],ctx->send_hdr->sizes[0]);
        if (!en_data)
            return 0; // 内存不足时，关闭当前句柄。

        /*追加到发送队列末尾。*/
        abcdk_tree_insert2(ctx->send_queue, en_data, 0);

        /*发送一次即可。*/
        ctx->send_hdr_ok = 1;
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

static int _abcdk_openssl_darknet_read_init(abcdk_openssl_darknet_t *ctx)
{
    abcdk_object_t *hdr;
    int ver;
    int chk;

    /*用RSA解密头部。*/
    hdr = abcdk_openssl_rsa_update(ctx->rsa_recv_ctx,ctx->recv_hdr->pptrs[0],ctx->recv_hdr->sizes[0],0);
    if(!hdr)
        return -1;

    /*解析头部。*/
    ver = abcdk_bloom_read_number(hdr->pptrs[0], hdr->sizes[0], 0, 8);
    memcpy(ctx->evp_recv_key, hdr->pptrs[0] + 1, 32);
    memcpy(ctx->evp_recv_iv, hdr->pptrs[0] + 1 + 32, 16);

    /*释放。*/
    abcdk_object_unref(&hdr);

    /*仅支持1版本。*/
    if (ver != 1)
        return -2;

    chk = EVP_CipherInit_ex(ctx->evp_recv_ctx, EVP_aes_256_ctr(), NULL, ctx->evp_recv_key, ctx->evp_recv_iv, 0);
    if (chk != 1)
        return -3;

    return 0;
}

static abcdk_object_t *_abcdk_openssl_darknet_read_update(abcdk_openssl_darknet_t *ctx, const void *in, int in_len)
{
    abcdk_object_t *de_data = NULL;
    int de_outlen;

    de_data = abcdk_object_alloc2(in_len);
    if (!de_data)
        return NULL; 

    EVP_CipherUpdate(ctx->evp_recv_ctx, de_data->pptrs[0], &de_outlen, in, in_len);
    assert(de_outlen == in_len);

    return de_data;
}


ssize_t abcdk_openssl_darknet_read(abcdk_openssl_darknet_t *ctx, void *data, size_t size)
{
    abcdk_object_t *de_data = NULL;
    ssize_t rlen = 0;
    int chk;

    assert(ctx != NULL && data != NULL && size > 0);

NEXT_LOOP:

    if (ctx->hdr_len == ctx->recv_hdr_len)
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

    /*先读取头部。*/
    if (ctx->recv_hdr_len < ctx->hdr_len)
    {
        /*计算待提取的头部长度并提取。*/
        size_t diff = ABCDK_MIN(ctx->hdr_len - ctx->recv_hdr_len, rlen);
        memcpy(ctx->recv_hdr->pptrs[0] + ctx->recv_hdr_len, ctx->recv_buf->pptrs[0], diff);
        ctx->recv_hdr_len += diff;

        /*当向量读取完整后，进行初始化。*/
        if (ctx->recv_hdr_len == ctx->hdr_len)
        {
            /*初始化接收环境。*/
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