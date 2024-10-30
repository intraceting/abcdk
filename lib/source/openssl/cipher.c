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

#ifdef HEADER_RSA_H

    /*RSA环境。*/
    RSA *rsa_ctx;

#endif // HEADER_RSA_H

    /*临时缓存。*/
    uint8_t tmpbuf[8192];

    /*明文块长度。*/
    int plaintext_bsize;

    /*密文块长度。*/
    int ciphertext_bsize;

}; // abcdk_openssl_cipher_t;

void _abcdk_openssl_cipher_rand_generate(uint8_t *buf, int len)
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

void abcdk_openssl_cipher_destroy(abcdk_openssl_cipher_t **ctx)
{
    abcdk_openssl_cipher_t *ctx_p;

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

    abcdk_object_unref(&ctx_p->evp_key);

#ifdef HEADER_RSA_H

    if (ctx_p->rsa_ctx)
        RSA_free(ctx_p->rsa_ctx);

#endif // HEADER_RSA_H

    abcdk_heap_free(ctx_p);
}

static int _abcdk_openssl_cipher_rsa_init(abcdk_openssl_cipher_t *ctx, const uint8_t *key, size_t key_len)
{
    FILE *fp = NULL;
    int chk;

#ifdef HEADER_RSA_H

    fp = fmemopen((void *)key, key_len, "r");
    if (!fp)
        return -1;

    if (ctx->scheme == ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PRIVATE)
    {
        ctx->rsa_ctx = PEM_read_RSAPrivateKey(fp, NULL, NULL, NULL);
    }
    else if (ctx->scheme == ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PUBLIC)
    {
        ctx->rsa_ctx = PEM_read_RSAPublicKey(fp, NULL, NULL, NULL);
    }

    fclose(fp);
    fp = NULL;

    if (!ctx->rsa_ctx)
        goto ERR;

    ctx->plaintext_bsize = RSA_size(ctx->rsa_ctx) - RSA_PKCS1_PADDING_SIZE;
    ctx->ciphertext_bsize = RSA_size(ctx->rsa_ctx);

    return 0;

ERR:

    if (fp)
        fclose(fp);

#endif // HEADER_RSA_H

    return -1;
}

static int _abcdk_openssl_cipher_rsa_update_fragment(abcdk_openssl_cipher_t *ctx, uint8_t *out, int out_max, const uint8_t *in, int in_len, int enc)
{
    int chk;

    /*
     * |DATA    |PADDING  |
     * |N bytes |11 bytes |
     */

#ifdef HEADER_RSA_H

    if (enc)
    {
        if (in_len > ctx->plaintext_bsize)
            return -1;

        if (out_max < ctx->ciphertext_bsize)
            return -2;

        memcpy(ctx->tmpbuf, in, in_len);
        _abcdk_openssl_cipher_rand_generate(ctx->tmpbuf + in_len, ctx->plaintext_bsize - in_len);

        if (ctx->scheme == ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PRIVATE)
            chk = RSA_private_encrypt(ctx->plaintext_bsize, ctx->tmpbuf, out, ctx->rsa_ctx, RSA_PKCS1_PADDING);
        else if (ctx->scheme == ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PUBLIC)
            chk = RSA_public_encrypt(ctx->plaintext_bsize, ctx->tmpbuf, out, ctx->rsa_ctx, RSA_PKCS1_PADDING);
        else
            chk = -1;

        if (chk <= 0)
            return -3;

        return ctx->ciphertext_bsize;
    }
    else
    {
        if (in_len != ctx->ciphertext_bsize)
            return -1;

        if (out_max < ctx->plaintext_bsize)
            return -2;

        if (ctx->scheme == ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PRIVATE)
            chk = RSA_private_decrypt(in_len, in, ctx->tmpbuf, ctx->rsa_ctx, RSA_PKCS1_PADDING);
        else if (ctx->scheme == ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PUBLIC)
            chk = RSA_public_decrypt(in_len, in, ctx->tmpbuf, ctx->rsa_ctx, RSA_PKCS1_PADDING);
        else
            chk = -1;

        if (chk <= 0)
            return -3;

        memcpy(out, ctx->tmpbuf, ctx->plaintext_bsize);

        return ctx->plaintext_bsize;
    }

#endif // #ifdef HEADER_RSA_H

    return -1;
}

abcdk_object_t *_abcdk_openssl_cipher_rsa_update(abcdk_openssl_cipher_t *ctx, const uint8_t *in, int in_len, int enc)
{
    abcdk_object_t *out;
    int blocks;
    int chk;

    if (enc)
    {
        blocks = abcdk_align(in_len, ctx->plaintext_bsize) / ctx->plaintext_bsize;

        out = abcdk_object_alloc2(blocks * ctx->ciphertext_bsize);
        if (!out)
            return NULL;

        for (int i = 0; i < blocks; i++)
        {
            if (i < (blocks - 1) || (in_len % ctx->plaintext_bsize) == 0)
            {
                _abcdk_openssl_cipher_rsa_update_fragment(ctx, out->pptrs[0] + i * ctx->ciphertext_bsize, ctx->ciphertext_bsize,
                                                  in + i * ctx->plaintext_bsize, ctx->plaintext_bsize, 1);
            }
            else
            {
                /*明文没有块对齐时，最后一块需要单独计算。*/
                _abcdk_openssl_cipher_rsa_update_fragment(ctx, out->pptrs[0] + i * ctx->ciphertext_bsize, ctx->ciphertext_bsize,
                                                  in + i * ctx->plaintext_bsize, in_len % ctx->plaintext_bsize, 1);
            }
        }
    }
    else
    {
        /*密文必须是块对齐的。*/
        if ((in_len % ctx->ciphertext_bsize) != 0)
            return NULL;

        blocks = in_len / ctx->ciphertext_bsize;

        out = abcdk_object_alloc2(blocks * ctx->plaintext_bsize);
        if (!out)
            return NULL;

        for (int i = 0; i < blocks; i++)
        {
            _abcdk_openssl_cipher_rsa_update_fragment(ctx, out->pptrs[0] + i * ctx->plaintext_bsize, ctx->plaintext_bsize,
                                              in + i * ctx->ciphertext_bsize, ctx->ciphertext_bsize, 0);
        }
    }

    return out;
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

    ctx->plaintext_bsize = 16 * 64;
    ctx->ciphertext_bsize = 16 * 64;

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

    if (scheme == ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PRIVATE || scheme == ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PUBLIC)
    {
        ctx->scheme = scheme;
        chk = _abcdk_openssl_cipher_rsa_init(ctx, key, key_len);
    }
    else if (scheme == ABCDK_OPENSSL_CIPHER_SCHEME_AES_256_GCM)
    {
        ctx->scheme = scheme;
        chk = _abcdk_openssl_cipher_aes256gcm_init(ctx, key, key_len);
    }
    else if (scheme == ABCDK_OPENSSL_CIPHER_SCHEME_AES_256_CBC)
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

    chk = _abcdk_openssl_cipher_init(ctx, scheme, key, key_len);
    if (chk == 0)
        return ctx;

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

    if (ctx->scheme == ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PRIVATE || ctx->scheme == ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PUBLIC)
        out = _abcdk_openssl_cipher_rsa_update(ctx, in, in_len, enc);
    else if (ctx->scheme == ABCDK_OPENSSL_CIPHER_SCHEME_AES_256_GCM)
        out = _abcdk_openssl_cipher_aes256gcm_update(ctx, in, in_len, enc);
    else if (ctx->scheme == ABCDK_OPENSSL_CIPHER_SCHEME_AES_256_CBC)
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
    assert(!enc || (enc && in_len < 16 * 1024 * 1024));

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
        old_crc32 = (uint32_t)abcdk_bloom_read_number(dst_p->pptrs[0], dst_p->sizes[0], 32, 32);
        new_crc32 = abcdk_crc32(dst_p->pptrs[0] + 8, dst_p->sizes[0] - 8);

        if (old_crc32 != new_crc32 || old_len != dst_p->sizes[0] - 8)
            goto ERR;

        /*跳过头部。*/
        dst_p->pptrs[0] += 8;
        dst_p->sizes[0] -= 8;
    }

    abcdk_object_unref(&src_p);
    return dst_p;

ERR:

    abcdk_object_unref(&src_p);
    abcdk_object_unref(&dst_p);
    return NULL;
}

#endif // OPENSSL_VERSION_NUMBER