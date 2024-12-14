/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/openssl/bio.h"

#ifdef HEADER_BIO_H

typedef struct _abcdk_openssl_BIO
{
    /**/
    BIO_METHOD *method_ctx;

    /*类型*/
    uint8_t type;
#define ABCDK_OPENSSL_BIO_DARKNET 1

    /*Darknet环境。*/
    abcdk_openssl_darknet_t *dkt_ctx;

} abcdk_openssl_BIO_t;


static BIO_METHOD *_abcdk_openssl_BIO_meth_new(int type, const char *name)
{
    BIO_METHOD *method_ctx;

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    method_ctx = (BIO_METHOD *)abcdk_heap_alloc(sizeof(BIO_METHOD));
    if (!method_ctx)
        return NULL;

    method_ctx->type = type;
    method_ctx->name = name;

#else
    method_ctx = BIO_meth_new(type, name);
    if (!method_ctx)
        return NULL;

#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

    return method_ctx;
}

static void _abcdk_openssl_BIO_meth_free(BIO_METHOD *method_ctx)
{
    if (!method_ctx)
        return;

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    abcdk_heap_free(method_ctx);
#else
    BIO_meth_free(method_ctx);
#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L
}

static int _abcdk_openssl_BIO_meth_set_write(BIO_METHOD *method_ctx,int (*write_cb)(BIO *, const char *, int))
{
    int chk;

    assert(method_ctx != NULL && write_cb != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    method_ctx->bwrite = write_cb;
    chk = 1;

#else
    chk = BIO_meth_set_write(method_ctx, write_cb);

#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

    return chk;
}

static int _abcdk_openssl_BIO_meth_set_read(BIO_METHOD *method_ctx,int (*read_cb)(BIO *, char *, int))
{
    int chk;

    assert(method_ctx != NULL && read_cb != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    method_ctx->bread = read_cb;
    chk = 1;

#else
    chk = BIO_meth_set_read(method_ctx, read_cb);

#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

    return chk;
}

static int _abcdk_openssl_BIO_meth_set_ctrl(BIO_METHOD *method_ctx, long (*ctrl_cb)(BIO *, int, long, void *))
{
    int chk;

    assert(method_ctx != NULL && ctrl_cb != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    method_ctx->ctrl = ctrl_cb;
    chk = 1;

#else
    chk = BIO_meth_set_ctrl(method_ctx, ctrl_cb);

#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

    return chk;
}

static int _abcdk_openssl_BIO_meth_set_create(BIO_METHOD *method_ctx, int (*create_cb)(BIO *))
{
    int chk;

    assert(method_ctx != NULL && create_cb != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    method_ctx->create = create_cb;
    chk = 1;

#else
    chk = BIO_meth_set_create(method_ctx, create_cb);

#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

    return chk;
}

static int _abcdk_openssl_BIO_meth_set_destroy(BIO_METHOD *method_ctx, int (*destroy_cb)(BIO *))
{
    int chk;

    assert(method_ctx != NULL && destroy_cb != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    method_ctx->destroy = destroy_cb;
    chk = 1;

#else
    chk = BIO_meth_set_destroy(method_ctx, destroy_cb);

#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

    return chk;
}

static void *_abcdk_openssl_BIO_get_data(BIO *bio_ctx)
{
    assert(bio_ctx != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    return bio_ctx->ptr;
#else
    return BIO_get_data(bio_ctx);
#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

}

static void _abcdk_openssl_BIO_set_data(BIO *bio_ctx, void *ptr)
{
    assert(bio_ctx != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    bio_ctx->ptr = ptr;
#else
    BIO_set_data(bio_ctx, ptr);
#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L
}


static int _abcdk_openssl_BIO_read_cb(BIO *bio_ctx, char *buf, int len)
{
    abcdk_openssl_BIO_t *bio_p = (abcdk_openssl_BIO_t *)_abcdk_openssl_BIO_get_data(bio_ctx);
    int rlen = 0;

    if(!(bio_p != NULL && bio_p->type == ABCDK_OPENSSL_BIO_DARKNET))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_READ, BIO_R_BROKEN_PIPE, __FUNCTION__, __LINE__);
        return -1;
    }

    if (!(buf != NULL && len > 0))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_READ, BIO_R_NULL_PARAMETER, __FUNCTION__, __LINE__);
        return -1;
    }

    ERR_clear_error(); /*清除历史错误记录，非常重要。*/
    BIO_clear_retry_flags(bio_ctx); 

    rlen = abcdk_openssl_darknet_read(bio_p->dkt_ctx, buf, len);
    if (rlen < 0)
    {
        BIO_set_retry_read(bio_ctx); /*设置重试标志，非常重要。*/
        return -1;
    }

    return rlen;
}

static int _abcdk_openssl_BIO_write_cb(BIO *bio_ctx, const char *buf, int len)
{
    abcdk_openssl_BIO_t *bio_p = (abcdk_openssl_BIO_t *)_abcdk_openssl_BIO_get_data(bio_ctx);
    int slen = 0;

    if(!(bio_p != NULL && bio_p->type == ABCDK_OPENSSL_BIO_DARKNET))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_WRITE, BIO_R_BROKEN_PIPE, __FUNCTION__, __LINE__);
        return -1;
    }

    if (!(buf != NULL && len > 0))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_WRITE, BIO_R_NULL_PARAMETER, __FUNCTION__, __LINE__);
        return -1;
    }

    ERR_clear_error(); /*清除历史错误记录，非常重要。*/
    BIO_clear_retry_flags(bio_ctx);
    
    slen = abcdk_openssl_darknet_write(bio_p->dkt_ctx, buf, len);
    if (slen < 0)
    {
        BIO_set_retry_write(bio_ctx); /*设置重试标志，非常重要。*/
        return -1;
    }

    return slen;
}

static long _abcdk_openssl_BIO_ctrl_cb(BIO *bio_ctx, int cmd, long num, void *ptr)
{
    abcdk_openssl_BIO_t *bio_p = (abcdk_openssl_BIO_t *)_abcdk_openssl_BIO_get_data(bio_ctx);
    int chk = 0;

    if (!(bio_p != NULL && bio_p->type == ABCDK_OPENSSL_BIO_DARKNET))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_CTRL, BIO_R_BROKEN_PIPE, __FUNCTION__, __LINE__);
        return 0;
    }

    ERR_clear_error(); /*清除历史错误记录，非常重要。*/

    switch (cmd)
    {
    case BIO_C_SET_FD:
    {
        int fd = ABCDK_PTR2I32(ptr, 0);
        if (fd >= 0)
        {
            abcdk_openssl_darknet_set_fd(bio_p->dkt_ctx, fd, 0);
            chk = 1;
        }
        else
        {
            chk = 0;
        }
    }
    break;
    case BIO_C_GET_FD:
    {
        ABCDK_PTR2I32(ptr, 0) = abcdk_openssl_darknet_get_fd(bio_p->dkt_ctx, 0);
        chk = 1;
    }
    break;
    default:
    {
        chk = 1;
    }
    break;
    }

    return chk;
}

static int _abcdk_openssl_BIO_create_cb(BIO *bio)
{
    int chk = 0;

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    bio->init = 1;
    bio->num = 0;
    bio->ptr = NULL;
    bio->flags = 0;

    chk = 1;
#else
    BIO_set_init(bio, 1);
    
    chk = 1;

#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

    return chk;
}

static int _abcdk_openssl_BIO_destroy_cb(BIO *bio_ctx)
{
    abcdk_openssl_BIO_t *bio_p = (abcdk_openssl_BIO_t *)_abcdk_openssl_BIO_get_data(bio_ctx);

    if(!bio_p)
        return 1;

    if(bio_p->type != ABCDK_OPENSSL_BIO_DARKNET)
        return 0;

    abcdk_openssl_darknet_destroy(&bio_p->dkt_ctx);
    _abcdk_openssl_BIO_meth_free(bio_p->method_ctx);
    abcdk_heap_free(bio_p);

    return 1;
}

void abcdk_openssl_BIO_destroy(BIO **bio_ctx)
{
    BIO *bio_ctx_p;

    if(!bio_ctx ||!*bio_ctx)
        return;

    bio_ctx_p = *bio_ctx;
    *bio_ctx = NULL;

    BIO_free(bio_ctx_p);
}

BIO *abcdk_openssl_BIO_s_Darknet(RSA *rsa_ctx, int use_pubkey)
{
    abcdk_openssl_BIO_t *bio_p;
    BIO *openssl_bio_p;

    assert(rsa_ctx != NULL);
    
    bio_p = (abcdk_openssl_BIO_t*)abcdk_heap_alloc(sizeof(abcdk_openssl_BIO_t));
    if (!bio_p)
        goto ERR;

    bio_p->type = ABCDK_OPENSSL_BIO_DARKNET;
    bio_p->dkt_ctx = abcdk_openssl_darknet_create(rsa_ctx,use_pubkey);
    bio_p->method_ctx = _abcdk_openssl_BIO_meth_new(BIO_TYPE_FD,"Darknet BIO");

    if (!bio_p->dkt_ctx || !bio_p->method_ctx)
        goto ERR;
    
    _abcdk_openssl_BIO_meth_set_write(bio_p->method_ctx,_abcdk_openssl_BIO_write_cb);
    _abcdk_openssl_BIO_meth_set_read(bio_p->method_ctx,_abcdk_openssl_BIO_read_cb);
    _abcdk_openssl_BIO_meth_set_ctrl(bio_p->method_ctx,_abcdk_openssl_BIO_ctrl_cb);
    _abcdk_openssl_BIO_meth_set_create(bio_p->method_ctx,_abcdk_openssl_BIO_create_cb);
    _abcdk_openssl_BIO_meth_set_destroy(bio_p->method_ctx,_abcdk_openssl_BIO_destroy_cb);

    openssl_bio_p = BIO_new(bio_p->method_ctx);
    if (!openssl_bio_p)
        goto ERR;

    /*关联到一起。*/
    _abcdk_openssl_BIO_set_data(openssl_bio_p, bio_p);

    /*关联成功后，清理野指针。*/
    bio_p = NULL;

    return openssl_bio_p;

ERR:

    if(openssl_bio_p)
        BIO_free(openssl_bio_p);

    if(bio_p)
    {
        abcdk_openssl_darknet_destroy(&bio_p->dkt_ctx);
        _abcdk_openssl_BIO_meth_free(bio_p->method_ctx);
        abcdk_heap_free(bio_p);
    }

    return NULL;
}

BIO *abcdk_openssl_BIO_s_Darknet_form_file(const char *rsa_file, int pubkey)
{
    BIO *bio_ctx;
    RSA *rsa_ctx;

    assert(rsa_file != NULL);
    
    rsa_ctx = abcdk_openssl_rsa_load(rsa_file, pubkey, NULL);
    if(!rsa_ctx)
        return NULL;

    bio_ctx = abcdk_openssl_BIO_s_Darknet(rsa_ctx, pubkey);
    abcdk_openssl_rsa_free(&rsa_ctx);

    if (!bio_ctx)
        return NULL;

    return bio_ctx;
}

#endif // HEADER_BIO_H
