/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/openssl/cipher.h"

#ifdef OPENSSL_VERSION_NUMBER

/**简单的加密接口。 */
struct _abcdk_cipher
{
    /*方案。*/
    int scheme;

    /*EVP密钥。*/
    abcdk_object_t *evp_key;

    /*EVP环境。*/
    EVP_CIPHER_CTX *evp_enc_ctx;
    EVP_CIPHER_CTX *evp_dec_ctx;

#ifdef HEADER_RSA_H

    /*RSA环境。*/
    RSA *rsa_ctx;

#endif // HEADER_RSA_H

    /*明文块大小。*/
    int plaintext_bsize;

    /*密文块大小。*/
    int ciphertext_bsize;

    /*临时缓存。*/
    uint8_t tmpbuf[8192];

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

    if (ctx_p->evp_enc_ctx)
    {
        EVP_CIPHER_CTX_free(ctx_p->evp_enc_ctx);
        ctx_p->evp_enc_ctx = NULL;
    }

    if (ctx_p->evp_dec_ctx)
    {
        EVP_CIPHER_CTX_free(ctx_p->evp_dec_ctx);
        ctx_p->evp_dec_ctx = NULL;
    }

    abcdk_object_unref(&ctx_p->evp_key);

#ifdef HEADER_RSA_H

    if (ctx_p->rsa_ctx)
        RSA_free(ctx_p->rsa_ctx);

#endif // HEADER_RSA_H

    abcdk_heap_free(ctx_p);
}

static int _abcdk_cipher_init_rsa_ecb(abcdk_cipher_t *ctx, const uint8_t *key, size_t key_len)
{
    FILE *fp = NULL;
    int chk;

#ifdef HEADER_RSA_H

    fp = fmemopen((void *)key, key_len, "r");
    if (!fp)
        return -1;

    if (ctx->scheme == ABCDK_CIPHER_SCHEME_RSA_PRIVATE)
    {
        ctx->rsa_ctx = PEM_read_RSAPrivateKey(fp, NULL, NULL, NULL);
    }
    else if (ctx->scheme == ABCDK_CIPHER_SCHEME_RSA_PUBLIC)
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

static int _abcdk_cipher_rsa_ecb_update(abcdk_cipher_t *ctx, uint8_t *out, int out_max, const uint8_t *in, int in_len, int enc)
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
        _abcdk_cipher_rand_generate(ctx->tmpbuf + in_len, ctx->plaintext_bsize - in_len);

        if (ctx->scheme == ABCDK_CIPHER_SCHEME_RSA_PRIVATE)
            chk = RSA_private_encrypt(ctx->plaintext_bsize, ctx->tmpbuf, out, ctx->rsa_ctx, RSA_PKCS1_PADDING);
        else if (ctx->scheme == ABCDK_CIPHER_SCHEME_RSA_PUBLIC)
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

        if (ctx->scheme == ABCDK_CIPHER_SCHEME_RSA_PRIVATE)
            chk = RSA_private_decrypt(in_len, in, ctx->tmpbuf, ctx->rsa_ctx, RSA_PKCS1_PADDING);
        else if (ctx->scheme == ABCDK_CIPHER_SCHEME_RSA_PUBLIC)
            chk = RSA_public_decrypt(in_len, in, ctx->tmpbuf, ctx->rsa_ctx, RSA_PKCS1_PADDING);
        else
            chk = -1;

        if (chk <= 0)
            return -3;

        memcpy(out,ctx->tmpbuf,ctx->plaintext_bsize);

        return ctx->plaintext_bsize;
    }

#endif // #ifdef HEADER_RSA_H

    return -1;
}

static int _abcdk_cipher_init_aes_256_ecb(abcdk_cipher_t *ctx, const uint8_t *key, size_t key_len)
{
    int chk;

    ctx->evp_key = abcdk_object_alloc2(32);
    if (!ctx->evp_key)
        return -1;

    abcdk_sha256_once(key, key_len, ctx->evp_key->pptrs[0]);

    ctx->evp_enc_ctx = EVP_CIPHER_CTX_new();
    ctx->evp_dec_ctx = EVP_CIPHER_CTX_new();

    if (!ctx->evp_enc_ctx || !ctx->evp_dec_ctx)
        return -3;

    chk = EVP_CipherInit_ex(ctx->evp_enc_ctx, EVP_aes_256_ecb(), NULL, ctx->evp_key->pptrs[0], NULL, 1);
    if (chk != 1)
        return -4;

    chk = EVP_CipherInit_ex(ctx->evp_dec_ctx, EVP_aes_256_ecb(), NULL, ctx->evp_key->pptrs[0], NULL, 0);
    if (chk != 1)
        return -5;

    EVP_CIPHER_CTX_set_padding(ctx->evp_enc_ctx, 0);
    EVP_CIPHER_CTX_set_padding(ctx->evp_dec_ctx, 0);

    ctx->plaintext_bsize = EVP_CIPHER_CTX_block_size(ctx->evp_enc_ctx) - 4;//4字节的盐。
    ctx->ciphertext_bsize = EVP_CIPHER_CTX_block_size(ctx->evp_dec_ctx);

    return 0;
}

static int _abcdk_cipher_aes_256_ecb_update(abcdk_cipher_t *ctx, uint8_t *out, int out_max, const uint8_t *in, int in_len, int enc)
{
    int block_len;
    int chk;

    /*
     * |DATA    |SALT    |
     * |N bytes |4 Bytes |
     */

    if (enc)
    {
        if (in_len > ctx->plaintext_bsize)
            return -1;

        if (out_max < ctx->ciphertext_bsize)
            return -2;

        memcpy(ctx->tmpbuf, in, in_len);
        _abcdk_cipher_rand_generate(ctx->tmpbuf + in_len, ctx->plaintext_bsize - in_len + 4);

        chk = abcdk_openssl_evp_cipher_update(ctx->evp_enc_ctx, out, ctx->tmpbuf, ctx->plaintext_bsize + 4);
        if (chk <= 0)
            return -2;

        return ctx->ciphertext_bsize;
    }
    else
    {
        /*密文的长度必须是块对齐的。*/
        if (in_len != ctx->ciphertext_bsize)
            return -1;

        if (out_max < ctx->plaintext_bsize)
            return -2;

        chk = abcdk_openssl_evp_cipher_update(ctx->evp_dec_ctx, ctx->tmpbuf, in, in_len);
        if (chk <= 0)
            return -2;

        memcpy(out,ctx->tmpbuf,ctx->plaintext_bsize);

        return ctx->plaintext_bsize;
    }

    return -1;
}

static int _abcdk_cipher_init(abcdk_cipher_t *ctx, int scheme, const uint8_t *key, size_t key_len)
{
    int chk;

    if (scheme == ABCDK_CIPHER_SCHEME_RSA_PRIVATE || scheme == ABCDK_CIPHER_SCHEME_RSA_PUBLIC)
    {
        ctx->scheme = scheme;
        chk = _abcdk_cipher_init_rsa_ecb(ctx, key, key_len);
    }
    else if (scheme == ABCDK_CIPHER_SCHEME_AES_256_ECB)
    {
        ctx->scheme = scheme;
        chk = _abcdk_cipher_init_aes_256_ecb(ctx, key, key_len);
    }
    else
    {
        chk = -22;
    }

    return chk;
}

abcdk_cipher_t *abcdk_cipher_create(int scheme, const uint8_t *key, size_t key_len)
{
    abcdk_cipher_t *ctx;
    int chk;

    assert(key != NULL && key_len > 0);

    ctx = (abcdk_cipher_t *)abcdk_heap_alloc(sizeof(abcdk_cipher_t));
    if (!ctx)
        return NULL;

    chk = _abcdk_cipher_init(ctx, scheme, key, key_len);
    if (chk == 0)
        return ctx;

    abcdk_cipher_destroy(&ctx);
    return NULL;
}

abcdk_cipher_t *abcdk_cipher_create_from_file(int scheme, const char *key_file)
{
    abcdk_object_t *key;
    abcdk_cipher_t *ctx;

    assert(key_file != NULL);

    key = abcdk_mmap_filename(key_file, 0, 0, 0, 0);
    if (!key)
        return NULL;

    ctx = abcdk_cipher_create(scheme, key->pptrs[0], key->sizes[0]);
    abcdk_object_unref(&key);

    return ctx;
}

int _abcdk_cipher_update(abcdk_cipher_t *ctx, uint8_t *out, int out_max, const uint8_t *in, int in_len, int enc)
{
    int chk;

    assert(ctx != NULL && in != NULL && in_len > 0);

    if (ctx->scheme == ABCDK_CIPHER_SCHEME_RSA_PRIVATE || ctx->scheme == ABCDK_CIPHER_SCHEME_RSA_PUBLIC)
        chk = _abcdk_cipher_rsa_ecb_update(ctx, out, out_max, in, in_len, enc);
    else if (ctx->scheme == ABCDK_CIPHER_SCHEME_AES_256_ECB)
        chk = _abcdk_cipher_aes_256_ecb_update(ctx, out, out_max, in, in_len, enc);
    else
        chk = -1;

    return chk;
}

abcdk_object_t *abcdk_cipher_update(abcdk_cipher_t *ctx, const uint8_t *in, int in_len, int enc)
{
    abcdk_object_t *out;
    int blocks;
    int chk;

    assert(ctx != NULL && in != NULL && in_len > 0);

    if (enc)
    {
        blocks = abcdk_align(in_len, ctx->plaintext_bsize) / ctx->plaintext_bsize;

        out = abcdk_object_alloc2(blocks * ctx->ciphertext_bsize);
        if (!out)
            return NULL;

        for (int i = 0; i < blocks; i++)
        {
            if(i < (blocks-1) || (in_len % ctx->plaintext_bsize) == 0 )
            {
                _abcdk_cipher_update(ctx, out->pptrs[0] + i * ctx->ciphertext_bsize, ctx->ciphertext_bsize,
                                 in + i * ctx->plaintext_bsize, ctx->plaintext_bsize,1);
            }
            else
            {
                /*明文没有块对齐时，最后一块需要单独计算。*/
                _abcdk_cipher_update(ctx, out->pptrs[0] + i * ctx->ciphertext_bsize, ctx->ciphertext_bsize,
                                     in + i * ctx->plaintext_bsize, in_len % ctx->plaintext_bsize, 1);
            }
        }
    }
    else
    {
        /*密文必须是块对齐的。*/
        if((in_len % ctx->ciphertext_bsize) != 0)
            return NULL;

        blocks = in_len / ctx->ciphertext_bsize;

        out = abcdk_object_alloc2(blocks * ctx->plaintext_bsize);
        if (!out)
            return NULL;

        for (int i = 0; i < blocks; i++)
        {
            _abcdk_cipher_update(ctx, out->pptrs[0] + i * ctx->plaintext_bsize, ctx->plaintext_bsize,
                                 in + i * ctx->ciphertext_bsize, ctx->ciphertext_bsize, 0);
        }
    }

    return out;
}

#endif // OPENSSL_VERSION_NUMBER