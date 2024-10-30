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
    /*魔法数，检测环境是否被篡改。*/
    uint32_t magic;
#define ABCDK_OPENSSL_BIO_MAGIC 123456789

    abcdk_maskssl_t *ssl_ctx;
    BIO_METHOD *method_ctx;
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

    if(!(bio_p != NULL && bio_p->magic == ABCDK_OPENSSL_BIO_MAGIC))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_READ, BIO_R_BROKEN_PIPE, __FUNCTION__, __LINE__);
        return -1;
    }

    if (!(buf != NULL && len > 0))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_READ, BIO_R_NULL_PARAMETER, __FUNCTION__, __LINE__);
        return -1;
    }

    rlen = abcdk_maskssl_read(bio_p->ssl_ctx, buf, len);
    if (rlen < 0)
        BIO_set_retry_read(bio_ctx); /*设置重试标志，非常重要。*/

    return rlen;
}

static int _abcdk_openssl_BIO_write_cb(BIO *bio_ctx, const char *buf, int len)
{
    abcdk_openssl_BIO_t *bio_p = (abcdk_openssl_BIO_t *)_abcdk_openssl_BIO_get_data(bio_ctx);
    int slen = 0;

    if(!(bio_p != NULL && bio_p->magic == ABCDK_OPENSSL_BIO_MAGIC))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_WRITE, BIO_R_BROKEN_PIPE, __FUNCTION__, __LINE__);
        return -1;
    }

    if (!(buf != NULL && len > 0))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_WRITE, BIO_R_NULL_PARAMETER, __FUNCTION__, __LINE__);
        return -1;
    }


    slen = abcdk_maskssl_write(bio_p->ssl_ctx, buf, len);
    if (slen < 0)
        BIO_set_retry_write(bio_ctx); /*设置重试标志，非常重要。*/

    return slen;
}

static long _abcdk_openssl_BIO_ctrl_cb(BIO *bio_ctx, int cmd, long num, void *ptr)
{
    abcdk_openssl_BIO_t *bio_p = (abcdk_openssl_BIO_t *)_abcdk_openssl_BIO_get_data(bio_ctx);
    int chk = 0;

    if(!(bio_p != NULL && bio_p->magic == ABCDK_OPENSSL_BIO_MAGIC))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_CTRL, BIO_R_BROKEN_PIPE, __FUNCTION__, __LINE__);
        return -1;
    }

    switch (cmd)
    {
    case BIO_C_SET_FD:
    {
        int fd = ABCDK_PTR2I32(ptr, 0);
        if (fd >= 0)
        {
            abcdk_maskssl_set_fd(bio_p->ssl_ctx, fd, 0);
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
        ABCDK_PTR2I32(ptr, 0) = abcdk_maskssl_get_fd(bio_p->ssl_ctx, 0);
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

    if(bio_p->magic != ABCDK_OPENSSL_BIO_MAGIC)
        return 0;

    abcdk_maskssl_destroy(&bio_p->ssl_ctx);
    _abcdk_openssl_BIO_meth_free(bio_p->method_ctx);
    abcdk_heap_free(bio_p);

    return 1;
}


int abcdk_openssl_BIO_set_fd(BIO *bio_ctx, int fd)
{
    abcdk_openssl_BIO_t *bio_p;

    bio_p = _abcdk_openssl_BIO_get_data(bio_ctx);
    if(!bio_p || bio_p->magic != ABCDK_OPENSSL_BIO_MAGIC)
        return -1;

    abcdk_maskssl_set_fd(bio_p->ssl_ctx,fd,0);

    return 0;
}

int abcdk_openssl_BIO_get_fd(BIO *bio_ctx)
{
    abcdk_openssl_BIO_t *bio_p;

    bio_p = _abcdk_openssl_BIO_get_data(bio_ctx);
    if(!bio_p || bio_p->magic != ABCDK_OPENSSL_BIO_MAGIC)
        return -1;

    return abcdk_maskssl_get_fd(bio_p->ssl_ctx,0);
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

BIO *abcdk_openssl_BIO_s_MaskSSL(int scheme, const uint8_t *key,size_t size)
{
    abcdk_openssl_BIO_t *bio_p;
    BIO *openssl_bio_p;

    assert(scheme != 0 && key != NULL);
    
    bio_p = (abcdk_openssl_BIO_t*)abcdk_heap_alloc(sizeof(abcdk_openssl_BIO_t));
    if (!bio_p)
        goto ERR;

    bio_p->magic = ABCDK_OPENSSL_BIO_MAGIC;
    bio_p->ssl_ctx = abcdk_maskssl_create(scheme,key,size);
    bio_p->method_ctx = _abcdk_openssl_BIO_meth_new(BIO_TYPE_SOURCE_SINK,"MaskSSL BIO");

    if (!bio_p->ssl_ctx || !bio_p->method_ctx)
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
        abcdk_maskssl_destroy(&bio_p->ssl_ctx);
        _abcdk_openssl_BIO_meth_free(bio_p->method_ctx);
        abcdk_heap_free(bio_p);
    }

    return NULL;
}

BIO *abcdk_openssl_BIO_s_MaskSSL_form_file(int scheme,const char *file)
{
    BIO *bio_ctx;
    abcdk_object_t *key;

    assert(scheme != 0 && file != NULL);

    key = abcdk_mmap_filename(file, 0, 0, 0, 0);
    if (!key)
        return NULL;

    bio_ctx = abcdk_openssl_BIO_s_MaskSSL(scheme,key->pptrs[0], key->sizes[0]);
    abcdk_object_unref(&key);
    if (!bio_ctx)
        return NULL;

    return bio_ctx;
}

#endif // HEADER_BIO_H
