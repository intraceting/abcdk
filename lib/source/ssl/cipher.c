/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/ssl/cipher.h"

#ifdef OPENSSL_VERSION_NUMBER

/**简单的加密接口。 */
struct _abcdk_cipher
{
    /*方案。*/
    int scheme;

    /*标志。0 解密，!加密。*/
    int encrypt;

    /*密钥。*/
    abcdk_object_t *key;

    /*盐。*/
    abcdk_object_t *salt;

    /*EVP环境。*/
    EVP_CIPHER_CTX *evp_ctx;

}; // abcdk_cipher_t;

void _abcdk_cipher_rand_generate(uint8_t *buf, int len)
{
    if (len <= 0)
        return;

#if 0
    abcdk_rand_string(buf,len,0);
#elif 1
    RAND_bytes(buf, len);
#else
    memset(buf, '#', len);
#endif
}

void abcdk_cipher_destroy(abcdk_cipher_t **ctx)
{
    abcdk_cipher_t *ctx_p;

    if (!ctx || !*ctx)
        ;
    return;

    ctx_p = *ctx;
    *ctx = NULL;

    if (ctx_p->evp_ctx)
    {
        EVP_CIPHER_CTX_free(ctx_p->evp_ctx);
        ctx_p->evp_ctx = NULL;
    }

    abcdk_object_unref(&ctx_p->key);
    abcdk_object_unref(&ctx_p->salt);
    abcdk_heap_free(ctx_p);
}

static int _abcdk_cipher_init_aes_256_cbc(abcdk_cipher_t *ctx, const uint8_t *key, size_t key_len, const uint8_t *salt, size_t salt_len, int enc)
{
    int chk;

    ctx->scheme = ABCDK_CIPHER_SCHEME_AES_256_CBC;
    ctx->encrypt = enc;

    ctx->key = abcdk_object_alloc2(32);
    if (!ctx->key)
        return -1;

    ctx->salt = abcdk_object_alloc2(32);
    if (!ctx->salt)
        return -2;

    abcdk_sha256_once(key, key_len, ctx->key->pptrs[0]);
    abcdk_sha256_once(salt, salt_len, ctx->salt->pptrs[0]);

    ctx->evp_ctx = EVP_CIPHER_CTX_new();
    if (!ctx->evp_ctx)
        return -3;

    chk = EVP_CipherInit_ex(ctx->evp_ctx, EVP_aes_256_cbc(), NULL, ctx->key->pptrs[0], ctx->salt->pptrs[0], ctx->encrypt);
    if (chk != 1)
        return -4;

    EVP_CIPHER_CTX_set_padding(ctx->evp_ctx, 0);
    // EVP_CIPHER_CTX_set_key_length(ctx->evp_ctx,32);

    return 0;
}

static abcdk_object_t *_abcdk_cipher_aes_256_cbc_update(abcdk_cipher_t *ctx, const uint8_t *in, int in_len)
{
    abcdk_object_t *dst = NULL, *src = NULL;
    int align_len;
    int block_len;
    int salt_len;
    int chk;

    /*
     * |SALT    |LEN     |DATA    |PADDING |
     * |N bytes |4 bytes |N bytes |N bytes |
     */

    block_len = EVP_CIPHER_CTX_block_size(ctx->evp_ctx);
    salt_len = block_len - 4;

    if (ctx->encrypt)
    {
        /*明文的长度不能超过阈值。*/
        if(in_len > INT32_MAX - block_len)
            return NULL;

        /*计算块对齐长度。*/
        align_len = abcdk_align(block_len + in_len, block_len);

        src = abcdk_object_alloc2(align_len);
        if (!src)
            return NULL;

        dst = abcdk_object_alloc2(align_len);
        if (!dst)
            goto ERR;

        _abcdk_cipher_rand_generate(src->pptrs[0], salt_len);
        abcdk_bloom_write_number(src->pptrs[0] + salt_len, 4, 0, 32, in_len);
        memcpy(src->pptrs[0] + block_len, in, in_len);
        _abcdk_cipher_rand_generate(src->pptrs[0] + block_len + in_len, align_len - (block_len + in_len));

        chk = abcdk_openssl_evp_cipher_update(ctx->evp_ctx, dst->pptrs[0], src->pptrs[0], src->sizes[0]);
        if (chk <= 0)
            goto ERR;

        dst->sizes[0] = chk;
    }
    else
    {
        /*密文的长度必须是块对齐的。*/
        if(in_len % block_len != 0)
            return NULL;

        dst = abcdk_object_alloc2(in_len);
        if (!dst)
            return NULL;

        chk = abcdk_openssl_evp_cipher_update(ctx->evp_ctx, dst->pptrs[0], in, in_len);
        if (chk <= 0)
            goto ERR;

        chk = (int)abcdk_bloom_read_number(dst->pptrs[0] + salt_len, 4, 0, 32);

        /*明文的长度必须在合理的区间内。*/
        if (chk <= 0 || chk > in_len - block_len)
            goto ERR;

        /*检查明文和密文的块对齐长度是否相同。*/
        if(in_len != abcdk_align(block_len + chk, block_len))
            goto ERR;

        /*修改指针到明文首部。*/
        dst->pptrs[0] = dst->pptrs[0] + block_len;
        /*修改长度为明文实际长度。*/
        dst->sizes[0] = chk;
    }

    abcdk_object_unref(&src);
    return dst;

ERR:

    abcdk_object_unref(&dst);
    abcdk_object_unref(&src);
    return NULL;
}

static int _abcdk_cipher_init(abcdk_cipher_t *ctx, int scheme, const uint8_t *key, size_t key_len, const uint8_t *salt, size_t salt_len, int enc)
{
    int chk;

    if (scheme == ABCDK_CIPHER_SCHEME_AES_256_CBC)
        chk = _abcdk_cipher_init_aes_256_cbc(ctx, key, key_len, salt, salt_len, enc);
    else
        chk = -22;

    return chk;
}

abcdk_cipher_t *abcdk_cipher_create(int scheme, const uint8_t *key, size_t key_len, const uint8_t *salt, size_t salt_len, int enc)
{
    abcdk_cipher_t *ctx;
    int chk;

    assert(key != NULL && key_len > 0 && salt != NULL && salt_len > 0);

    ctx = (abcdk_cipher_t *)abcdk_heap_alloc(sizeof(abcdk_cipher_t));
    if (!ctx)
        return NULL;

    chk = _abcdk_cipher_init(ctx, scheme, key, key_len, salt, salt_len, enc);
    if (chk == 0)
        return ctx;

    abcdk_cipher_destroy(&ctx);
    return NULL;
}

abcdk_object_t *abcdk_cipher_update(abcdk_cipher_t *ctx, const uint8_t *in, int in_len)
{
    abcdk_object_t *out = NULL;

    assert(ctx != NULL && in != NULL && in_len > 0);

    if (ctx->scheme == ABCDK_CIPHER_SCHEME_AES_256_CBC)
        out = _abcdk_cipher_aes_256_cbc_update(ctx, in, in_len);
    else
        out = NULL;

    return out;
}

#endif // OPENSSL_VERSION_NUMBER