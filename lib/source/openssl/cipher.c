/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/openssl/cipher.h"

#ifdef OPENSSL_VERSION_NUMBER

/**简单的加密接口。 */
struct _abcdk_openssl_cipher
{
    /*方案。*/
    int scheme;

    /*EVP密钥。*/
    abcdk_object_t *evp_key;

    /*EVP向量。*/
    uint8_t evp_iv[16];

    /*EVP标签。*/
    uint8_t evp_tag[16];

    /*EVP环境。*/
    EVP_CIPHER_CTX *evp_ctx;

    /*临时缓存。*/
    uint8_t tmpbuf[8192];

    /*同步锁。*/
    abcdk_spinlock_t *locker_ctx;

}; // abcdk_openssl_cipher_t;

void _abcdk_openssl_cipher_rand_generate(uint8_t *buf, int len)
{
    if (len <= 0)
        return;

#if 0
    abcdk_rand_bytes(buf,len,0);
#elif 1
    RAND_bytes(buf, len);
#else
    memset(buf, '#', len);
#endif
}

void abcdk_openssl_cipher_destroy(abcdk_openssl_cipher_t **ctx)
{
    abcdk_openssl_cipher_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    abcdk_openssl_evp_cipher_ctx_free(&ctx_p->evp_ctx);
    abcdk_object_unref(&ctx_p->evp_key);
    abcdk_spinlock_destroy(&ctx_p->locker_ctx);
    abcdk_heap_free(ctx_p);
}

static int _abcdk_openssl_cipher_aes256gcm_init(abcdk_openssl_cipher_t *ctx, const uint8_t *key, size_t key_len)
{
    int chk;

    ctx->evp_key = abcdk_object_alloc2(32);
    if (!ctx->evp_key)
        return -1;

    abcdk_sha256_once(key, key_len, ctx->evp_key->pptrs[0]);

    ctx->evp_ctx = EVP_CIPHER_CTX_new();
    if (!ctx->evp_ctx)
        return -2;

    return 0;
}

static int _abcdk_openssl_cipher_aes256gcm_config(abcdk_openssl_cipher_t *ctx, int enc)
{
    int chk;

    EVP_CIPHER_CTX_cleanup(ctx->evp_ctx);

    chk = EVP_CipherInit_ex(ctx->evp_ctx, EVP_aes_256_gcm(), NULL, NULL, NULL, enc);
    if (chk != 1)
        return -1;

    chk = EVP_CIPHER_CTX_ctrl(ctx->evp_ctx, EVP_CTRL_GCM_SET_IVLEN, 16, NULL);
    if (chk != 1)
        return -2;

    chk = EVP_CipherInit_ex(ctx->evp_ctx, NULL, NULL, ctx->evp_key->pptrs[0], ctx->evp_iv, enc);
    if (chk != 1)
        return -3;

    return 0;
}

static int _abcdk_openssl_cipher_aes256gcm_update_fragment(abcdk_openssl_cipher_t *ctx, uint8_t *out, int out_max, const uint8_t *in, int in_len, int enc)
{
    int alen = 0, tlen = 0;
    int chk;

    if (out_max < in_len)
        return -1;

    chk = EVP_CipherUpdate(ctx->evp_ctx, out, &tlen, in, in_len);
    if (chk != 1)
        return -2;

    alen += tlen;

    /*解密时标签要在完成前设置。*/
    if (!enc)
    {
        chk = EVP_CIPHER_CTX_ctrl(ctx->evp_ctx, EVP_CTRL_GCM_SET_TAG, 16, ctx->evp_tag);
        if (chk != 1)
            return -4;
    }

    chk = EVP_CipherFinal(ctx->evp_ctx, out + alen, &tlen);
    if (chk != 1)
        return -3;

    alen += tlen;

    /*加密时标签要在完成后获取。*/
    if (enc)
    {
        chk = EVP_CIPHER_CTX_ctrl(ctx->evp_ctx, EVP_CTRL_GCM_GET_TAG, 16, ctx->evp_tag);
        if (chk != 1)
            return -4;
    }

    return alen;
}

abcdk_object_t *_abcdk_openssl_cipher_aes256gcm_update(abcdk_openssl_cipher_t *ctx, const uint8_t *in, int in_len, int enc)
{
    abcdk_object_t *out;
    int chk;

    /*
     * |DATA    |IV       |TAG      |
     * |N bytes |16 bytes |16 bytes |
     */

    if (enc)
    {
        /*生成随机IV。由于它将以明文方式进行传递，因此每次加密前必须重新生成。*/
        _abcdk_openssl_cipher_rand_generate(ctx->evp_iv, 16);

        chk = _abcdk_openssl_cipher_aes256gcm_config(ctx, 1);
        if (chk != 0)
            return NULL;

        out = abcdk_object_alloc2(in_len + 32);
        if (!out)
            return NULL;

        chk = _abcdk_openssl_cipher_aes256gcm_update_fragment(ctx, out->pptrs[0], out->sizes[0], in, in_len, 1);
        if (chk != in_len)
            goto ERR;

        /*在密文的末尾添加IV和TAG。*/
        memcpy(out->pptrs[0] + in_len, ctx->evp_iv, 16);
        memcpy(out->pptrs[0] + in_len + 16, ctx->evp_tag, 16);
    }
    else
    {
        if (in_len <= 32)
            return NULL;

        /*提取密文末尾的IV和TAG。*/
        memcpy(ctx->evp_iv, in + in_len - 32, 16);
        memcpy(ctx->evp_tag, in + in_len - 16, 16);

        chk = _abcdk_openssl_cipher_aes256gcm_config(ctx, 0);
        if (chk != 0)
            return NULL;

        out = abcdk_object_alloc2(in_len - 32);
        if (!out)
            return NULL;

        chk = _abcdk_openssl_cipher_aes256gcm_update_fragment(ctx, out->pptrs[0], out->sizes[0], in, in_len - 32, 0);
        if (chk != in_len - 32)
            goto ERR;
    }

    return out;

ERR:

    abcdk_object_unref(&out);
    return NULL;
}

static int _abcdk_openssl_cipher_aes256cbc_init(abcdk_openssl_cipher_t *ctx, const uint8_t *key, size_t key_len)
{
    int chk;

    ctx->evp_key = abcdk_object_alloc2(32);
    if (!ctx->evp_key)
        return -1;

    abcdk_sha256_once(key, key_len, ctx->evp_key->pptrs[0]);

    ctx->evp_ctx = EVP_CIPHER_CTX_new();
    if (!ctx->evp_ctx)
        return -2;

    return 0;
}

static int _abcdk_openssl_cipher_aes256cbc_config(abcdk_openssl_cipher_t *ctx, int enc)
{
    int chk;

    EVP_CIPHER_CTX_cleanup(ctx->evp_ctx);

    chk = EVP_CipherInit_ex(ctx->evp_ctx, EVP_aes_256_cbc(), NULL, NULL, NULL, enc);
    if (chk != 1)
        return -1;

    chk = EVP_CipherInit_ex(ctx->evp_ctx, NULL, NULL, ctx->evp_key->pptrs[0], ctx->evp_iv, enc);
    if (chk != 1)
        return -3;

    EVP_CIPHER_CTX_set_padding(ctx->evp_ctx, 0);

    return 0;
}

static int _abcdk_openssl_cipher_aes256cbc_update_fragment(abcdk_openssl_cipher_t *ctx, uint8_t *out, int out_max, const uint8_t *in, int in_len, int enc)
{
    int align_bsize;
    int blocks;
    int alen = 0, tlen = 0;
    int chk;

    align_bsize = abcdk_align(in_len, 16);

    if (enc)
    {
        if (out_max < align_bsize)
            return -1;

        blocks = in_len / 16;

        if (blocks > 0)
        {
            chk = EVP_CipherUpdate(ctx->evp_ctx, out, &tlen, in, blocks*16);
            if (chk != 1)
                return -2;

            alen += tlen;
        }

        if (in_len % 16 != 0)
        {
            memcpy(ctx->tmpbuf, in + blocks * 16, in_len % 16);
            _abcdk_openssl_cipher_rand_generate(ctx->tmpbuf + (in_len % 16), 16 - (in_len % 16));

            chk = EVP_CipherUpdate(ctx->evp_ctx, out + blocks * 16, &tlen, ctx->tmpbuf, 16);
            if (chk != 1)
                return -2;

            alen += tlen;
        }
    }
    else
    {
        if (in_len != align_bsize)
            return -1;

        if (out_max < align_bsize)
            return -2;

        chk = EVP_CipherUpdate(ctx->evp_ctx, out, &tlen, in, in_len);
        if (chk != 1)
            return -3;

        alen += tlen;
    }

    chk = EVP_CipherFinal(ctx->evp_ctx, out + alen, &tlen);
    if (chk != 1)
        return -4;

    alen += tlen;

    return alen;
}

abcdk_object_t *_abcdk_openssl_cipher_aes256cbc_update(abcdk_openssl_cipher_t *ctx, const uint8_t *in, int in_len, int enc)
{
    abcdk_object_t *out;
    int align_bsize;
    int blocks;
    int chk;

    /*
     * |DATA    |IV       |
     * |N bytes |16 bytes |
     */

    if (enc)
    {
        align_bsize = abcdk_align(in_len, 16);

        /*生成随机IV。由于它将以明文方式进行传递，因此每次加密前必须重新生成。*/
        _abcdk_openssl_cipher_rand_generate(ctx->evp_iv, 16);

        chk = _abcdk_openssl_cipher_aes256cbc_config(ctx, 1);
        if (chk != 0)
            return NULL;

        out = abcdk_object_alloc2(align_bsize + 16);
        if (!out)
            return NULL;

        chk = _abcdk_openssl_cipher_aes256cbc_update_fragment(ctx, out->pptrs[0], out->sizes[0], in, in_len, 1);
        if (chk != align_bsize)
            goto ERR;

        /*在密文的末尾添加IV。*/
        memcpy(out->pptrs[0] + align_bsize, ctx->evp_iv, 16);
    }
    else
    {
        if (in_len <= 16)
            return NULL;

        align_bsize = abcdk_align(in_len - 16, 16);

        /*密文必须是块对齐的。*/
        if ((align_bsize % 16) != 0)
            return NULL;

        /*提取密文末尾的IV。*/
        memcpy(ctx->evp_iv, in + align_bsize, 16);

        chk = _abcdk_openssl_cipher_aes256cbc_config(ctx, 0);
        if (chk != 0)
            return NULL;

        out = abcdk_object_alloc2(align_bsize);
        if (!out)
            return NULL;

        chk = _abcdk_openssl_cipher_aes256cbc_update_fragment(ctx, out->pptrs[0], out->sizes[0], in, align_bsize, 0);
        if (chk != align_bsize)
            goto ERR;
    }

    return out;

ERR:

    abcdk_object_unref(&out);
    return out;
}

static int _abcdk_openssl_cipher_init(abcdk_openssl_cipher_t *ctx, int scheme, const uint8_t *key, size_t key_len)
{
    int chk;

    if (scheme == ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM)
    {
        ctx->scheme = scheme;
        chk = _abcdk_openssl_cipher_aes256gcm_init(ctx, key, key_len);
    }
    else if (scheme == ABCDK_OPENSSL_CIPHER_SCHEME_AES256CBC)
    {
        ctx->scheme = scheme;
        chk = _abcdk_openssl_cipher_aes256cbc_init(ctx, key, key_len);
    }
    else
    {
        chk = -22;
    }

    return chk;
}

abcdk_openssl_cipher_t *abcdk_openssl_cipher_create(int scheme, const uint8_t *key, size_t key_len)
{
    abcdk_openssl_cipher_t *ctx;
    int chk;

    assert(key != NULL && key_len > 0);

    ctx = (abcdk_openssl_cipher_t *)abcdk_heap_alloc(sizeof(abcdk_openssl_cipher_t));
    if (!ctx)
        return NULL;

    ctx->locker_ctx = abcdk_spinlock_create();
    if(!ctx->locker_ctx)
        goto ERR;

    chk = _abcdk_openssl_cipher_init(ctx, scheme, key, key_len);
    if (chk != 0)
        goto ERR;

    /*OK.*/
    return ctx;

ERR:

    abcdk_openssl_cipher_destroy(&ctx);
    return NULL;
}

abcdk_openssl_cipher_t *abcdk_openssl_cipher_create_from_file(int scheme, const char *key_file)
{
    abcdk_object_t *key;
    abcdk_openssl_cipher_t *ctx;

    assert(key_file != NULL);

    key = abcdk_mmap_filename(key_file, 0, 0, 0, 0);
    if (!key)
        return NULL;

    ctx = abcdk_openssl_cipher_create(scheme, key->pptrs[0], key->sizes[0]);
    abcdk_object_unref(&key);

    return ctx;
}

abcdk_object_t *abcdk_openssl_cipher_update(abcdk_openssl_cipher_t *ctx, const uint8_t *in, int in_len, int enc)
{
    abcdk_object_t *out;

    assert(ctx != NULL && in != NULL && in_len > 0);

    if (ctx->scheme == ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM)
        out = _abcdk_openssl_cipher_aes256gcm_update(ctx, in, in_len, enc);
    else if (ctx->scheme == ABCDK_OPENSSL_CIPHER_SCHEME_AES256CBC)
        out = _abcdk_openssl_cipher_aes256cbc_update(ctx, in, in_len, enc);
    else
        out = NULL;

    return out;
}

abcdk_object_t *abcdk_openssl_cipher_update_pack(abcdk_openssl_cipher_t *ctx, const uint8_t *in, int in_len, int enc)
{
    abcdk_object_t *src_p = NULL;
    abcdk_object_t *dst_p = NULL;
    uint32_t old_len = 0;
    uint32_t old_crc32 = 0,new_crc32 = ~0;

    assert(ctx != NULL && in != NULL && in_len > 0);
    assert(!enc || (enc && in_len <= 16777215));

    /*
     * |Length  |CRC32   |Data    |
     * |4 Bytes |4 Bytes |N Bytes |
     *
     * Length：明文长度。
     * CRC32：校验码。
     * Data: 明文数据。
     */

    if(enc)
    {
        src_p = abcdk_object_alloc2(4 + 4 + in_len);
        if (!src_p)
            goto ERR;

        abcdk_bloom_write_number(src_p->pptrs[0], src_p->sizes[0], 0, 32, in_len);
        abcdk_bloom_write_number(src_p->pptrs[0], src_p->sizes[0], 32, 32, abcdk_crc32(in, in_len));
        memcpy(src_p->pptrs[0] + 8, in, in_len);

        dst_p = abcdk_openssl_cipher_update(ctx,src_p->pptrs[0],src_p->sizes[0],1);
        if(!dst_p)
            goto ERR;
    }
    else
    {
        dst_p = abcdk_openssl_cipher_update(ctx,in,in_len,0);
        if(!dst_p)
            goto ERR;

        if (dst_p->sizes[0] < 8)
            goto ERR;

        old_len = (uint32_t)abcdk_bloom_read_number(dst_p->pptrs[0], dst_p->sizes[0], 0, 32);

        if (old_len > dst_p->sizes[0] - 8)
            goto ERR;

        old_crc32 = (uint32_t)abcdk_bloom_read_number(dst_p->pptrs[0], dst_p->sizes[0], 32, 32);
        new_crc32 = abcdk_crc32(dst_p->pptrs[0] + 8, old_len);

        if (old_crc32 != new_crc32)
            goto ERR;

        /*跳过头部，指向数据区。*/
        dst_p->pptrs[0] += 8;
        dst_p->sizes[0] = old_len;
    }

    abcdk_object_unref(&src_p);
    return dst_p;

ERR:

    abcdk_object_unref(&src_p);
    abcdk_object_unref(&dst_p);
    return NULL;
}

void abcdk_openssl_cipher_lock(abcdk_openssl_cipher_t *ctx)
{
    assert(ctx != NULL);

    abcdk_spinlock_lock(ctx->locker_ctx,1);
}

int abcdk_openssl_cipher_unlock(abcdk_openssl_cipher_t *ctx,int exitcode)
{
    assert(ctx != NULL);

    abcdk_spinlock_unlock(ctx->locker_ctx);

    return exitcode;
}

#endif // OPENSSL_VERSION_NUMBER