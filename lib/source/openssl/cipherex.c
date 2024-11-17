/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/openssl/cipherex.h"

#ifdef OPENSSL_VERSION_NUMBER

/**简单的加密接口。 */
struct _abcdk_openssl_cipherex
{
    /*魔法数。*/
    uint32_t magic;
#define ABCDK_OPENSSL_CIPHEREX_MAGIC 123456789

    /*负截均衡游标。*/
    volatile uint64_t lb_pos;

    /*密钥组环境。*/
    abcdk_object_t *cipher_gp;

} ;//abcdk_openssl_cipherex_t;


void abcdk_openssl_cipherex_destroy(abcdk_openssl_cipherex_t **ctx)
{
    abcdk_openssl_cipherex_t *ctx_p;

    if(!ctx ||!*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    for(int i = 0;i<ctx_p->cipher_gp->numbers;i++)
    {
        if(!ctx_p->cipher_gp->pptrs[i])
            break;

        abcdk_openssl_cipher_destroy((abcdk_openssl_cipher_t **)&ctx_p->cipher_gp->pptrs[i]);
    }
}

abcdk_openssl_cipherex_t *abcdk_openssl_cipherex_create(int worker, int scheme, const uint8_t *key, size_t klen)
{
    abcdk_openssl_cipherex_t *ctx;

    assert(worker >0 && scheme > 0 && key != NULL &&  klen > 0);
    assert(scheme == ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PRIVATE ||
           scheme == ABCDK_OPENSSL_CIPHER_SCHEME_RSA_PUBLIC ||
           scheme == ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM ||
           scheme == ABCDK_OPENSSL_CIPHER_SCHEME_AES256CBC);

    ctx = (abcdk_openssl_cipherex_t*)abcdk_heap_alloc(sizeof(abcdk_openssl_cipherex_t));
    if(!ctx)
        return NULL;

    ctx->magic = ABCDK_OPENSSL_CIPHEREX_MAGIC;
    ctx->lb_pos = 0;

    ctx->cipher_gp = abcdk_object_alloc3(0,worker);
    if(!ctx->cipher_gp)
        goto ERR;

    for (int i = 0; i < ctx->cipher_gp->numbers; i++)
    {
        ctx->cipher_gp->pptrs[i] = (uint8_t *)abcdk_openssl_cipher_create(scheme, key, klen);
        if (!ctx->cipher_gp->pptrs[i])
            goto ERR;
    }

    return ctx;

ERR:

    abcdk_openssl_cipherex_destroy(&ctx);
    return NULL;
}

uint64_t _abcdk_openssl_cipherex_pos(abcdk_openssl_cipherex_t *ctx)
{
    return abcdk_atomic_fetch_and_add(&ctx->lb_pos,1) % (uint64_t)ctx->cipher_gp->numbers;
}

abcdk_object_t *abcdk_openssl_cipherex_update(abcdk_openssl_cipherex_t *ctx, const uint8_t *in, int in_len, int enc)
{
    abcdk_openssl_cipher_t *cipher_p = NULL;
    abcdk_object_t *dst_p = NULL;
    uint64_t pos = 0;

    assert(ctx != NULL && in != NULL && in_len > 0);
    assert(ctx->magic == ABCDK_OPENSSL_CIPHEREX_MAGIC);

    pos = _abcdk_openssl_cipherex_pos(ctx);
    cipher_p = (abcdk_openssl_cipher_t *)ctx->cipher_gp->pptrs[pos];

    abcdk_openssl_cipher_lock(cipher_p);
    dst_p = abcdk_openssl_cipher_update(cipher_p,in,in_len,enc);
    abcdk_openssl_cipher_unlock(cipher_p,0);

    return dst_p;
}


abcdk_object_t *abcdk_openssl_cipherex_update_pack(abcdk_openssl_cipherex_t *ctx, const uint8_t *in, int in_len, int enc)
{
    abcdk_openssl_cipher_t *cipher_p = NULL;
    abcdk_object_t *dst_p = NULL;
    
    uint64_t pos = 0;

    assert(ctx != NULL && in != NULL && in_len > 0);
    assert(ctx->magic == ABCDK_OPENSSL_CIPHEREX_MAGIC);

    pos = _abcdk_openssl_cipherex_pos(ctx);
    cipher_p = (abcdk_openssl_cipher_t *)ctx->cipher_gp->pptrs[pos];

    abcdk_openssl_cipher_lock(cipher_p);
    dst_p = abcdk_openssl_cipher_update_pack(cipher_p,in,in_len,enc);
    abcdk_openssl_cipher_unlock(cipher_p,0);

    return dst_p;
}

#endif //OPENSSL_VERSION_NUMBER