/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#include "abcdk/util/md5.h"

#ifdef HAVE_OPENSSL
#include <openssl/md5.h>
#endif //HAVE_OPENSSL

#ifdef HAVE_FFMPEG
#include <libavutil/avutil.h>
#include <libavutil/md5.h>
#endif //HAVE_FFMPEG

/** 简单的MD5。*/
typedef struct _abcdk_md5
{
#if defined(HEADER_MD5_H) && !defined(OPENSSL_NO_MD5)
    MD5_CTX ssl;
#elif defined(AVUTIL_MD5_H)
    struct AVMD5 *ff;
#else 
    int a;
#endif 
} abcdk_md5_t;

void abcdk_md5_destroy(abcdk_md5_t **ctx)
{
    abcdk_md5_t *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

#if defined(HEADER_MD5_H) && !defined(OPENSSL_NO_MD5)
    //
#elif defined(AVUTIL_MD5_H)
    if(ctx_p->ff)
    {
        av_free(ctx_p->ff);
        ctx_p->ff = NULL;
    }
#else 
    //
#endif 

    abcdk_heap_free(ctx_p);
}

abcdk_md5_t *abcdk_md5_create()
{
    abcdk_md5_t *ctx = NULL;

    ctx = abcdk_heap_alloc(sizeof(abcdk_md5_t));
    if(!ctx)
        return NULL;

#if defined(HEADER_MD5_H) && !defined(OPENSSL_NO_MD5)
    MD5_Init(&ctx->ssl);
#elif defined(AVUTIL_MD5_H)
    ctx->ff = av_md5_alloc();
    if(!ctx->ff)
        goto final_error;
    
    av_md5_init(ctx->ff);
#else 
    goto final_error;
#endif 

    return ctx;

final_error:

    abcdk_md5_destroy(&ctx);
    return NULL;
}

void abcdk_md5_reset(abcdk_md5_t *ctx)
{
    assert(ctx != NULL);

#if defined(HEADER_MD5_H) && !defined(OPENSSL_NO_MD5)
    MD5_Init(&ctx->ssl);
#elif defined(AVUTIL_MD5_H)
    av_md5_init(ctx->ff);
#else 
    //
#endif 
}

void abcdk_md5_update(abcdk_md5_t *ctx, const void *data, size_t size)
{
    assert(ctx != NULL && data != NULL);
    
#if defined(HEADER_MD5_H) && !defined(OPENSSL_NO_MD5)
    MD5_Update(&ctx->ssl,data,size);
#elif defined(AVUTIL_MD5_H)
    av_md5_update(ctx->ff,data,size);
#else 
    //
#endif 
}

void abcdk_md5_final(abcdk_md5_t *ctx,uint8_t hashcode[16])
{
    assert(ctx != NULL && hashcode != NULL);
        
#if defined(HEADER_MD5_H) && !defined(OPENSSL_NO_MD5)
    MD5_Final(hashcode,&ctx->ssl);
#elif defined(AVUTIL_MD5_H)
    av_md5_final(ctx->ff,hashcode);
#else 
    //
#endif 
}

void abcdk_md5_final2hex(abcdk_md5_t *ctx, char hashcode[33],int ABC)
{
    char buf[16];
    abcdk_md5_final(ctx, buf);
    abcdk_bin2hex(hashcode, buf, 16, ABC);
}