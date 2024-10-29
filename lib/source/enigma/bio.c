/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/enigma/bio.h"

#ifdef HEADER_BIO_H

typedef struct _abcdk_enigma_BIO
{
    /*魔法数，检测环境是否被篡改。*/
    uint32_t magic;
#define ABCDK_ENIGMA_BIO_MAGIC 123456789

    abcdk_enigma_ssl_t *enigma_ssl;
    BIO_METHOD *method;
} abcdk_enigma_BIO_t;


static BIO_METHOD *_abcdk_enigma_BIO_meth_new(int type, const char *name)
{
    BIO_METHOD *biom;

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    biom = (BIO_METHOD *)abcdk_heap_alloc(sizeof(BIO_METHOD));
    if (!biom)
        return NULL;

    biom->type = type;
    biom->name = name;

#else
    biom = BIO_meth_new(type, name);
    if (!biom)
        return NULL;

#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

    return biom;
}

static void _abcdk_enigma_BIO_meth_free(BIO_METHOD *biom)
{
    if (!biom)
        return;
#if OPENSSL_VERSION_NUMBER < 0x10100000L
    abcdk_heap_free(biom);
#else
    BIO_meth_free(biom);
#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L
}

static int _abcdk_enigma_BIO_meth_set_write(BIO_METHOD *biom,int (*write_cb)(BIO *, const char *, int))
{
    int chk;
    assert(biom != NULL && write_cb != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    biom->bwrite = write_cb;
    chk = 1;

#else
    chk = BIO_meth_set_write(biom, write_cb);

#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

    return chk;
}

static int _abcdk_enigma_BIO_meth_set_read(BIO_METHOD *biom,int (*read_cb)(BIO *, char *, int))
{
    int chk;
    assert(biom != NULL && read_cb != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    biom->bread = read_cb;
    chk = 1;

#else
    chk = BIO_meth_set_read(biom, read_cb);

#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

    return chk;
}

static int _abcdk_enigma_BIO_meth_set_ctrl(BIO_METHOD *biom, long (*ctrl_cb)(BIO *, int, long, void *))
{
    int chk;
    assert(biom != NULL && ctrl_cb != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    biom->ctrl = ctrl_cb;
    chk = 1;

#else
    chk = BIO_meth_set_ctrl(biom, ctrl_cb);

#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

    return chk;
}

static int _abcdk_enigma_BIO_meth_set_create(BIO_METHOD *biom, int (*create_cb)(BIO *))
{
    int chk;
    assert(biom != NULL && create_cb != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    biom->create = create_cb;
    chk = 1;

#else
    chk = BIO_meth_set_create(biom, create_cb);

#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

    return chk;
}

static int _abcdk_enigma_BIO_meth_set_destroy(BIO_METHOD *biom, int (*destroy_cb)(BIO *))
{
    int chk;
    assert(biom != NULL && destroy_cb != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    biom->destroy = destroy_cb;
    chk = 1;

#else
    chk = BIO_meth_set_destroy(biom, destroy_cb);

#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

    return chk;
}

static void *_abcdk_enigma_BIO_get_data(BIO *bio)
{
    assert(bio != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    return bio->ptr;
#else
    return BIO_get_data(bio);
#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L

}

static void _abcdk_enigma_BIO_set_data(BIO *bio, void *ptr)
{
    assert(bio != NULL);

#if OPENSSL_VERSION_NUMBER < 0x10100000L
    bio->ptr = ptr;
#else
    BIO_set_data(bio, ptr);
#endif // #if OPENSSL_VERSION_NUMBER < 0x10100000L
}


static int _abcdk_enigma_BIO_read_cb(BIO *bio, char *buf, int len)
{
    abcdk_enigma_BIO_t *bio_p = (abcdk_enigma_BIO_t *)_abcdk_enigma_BIO_get_data(bio);
    int rlen = 0;

    if(!(bio_p != NULL && bio_p->magic == ABCDK_ENIGMA_BIO_MAGIC))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_READ, BIO_R_BROKEN_PIPE, __FUNCTION__, __LINE__);
        return -1;
    }

    if (!(buf != NULL && len > 0))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_READ, BIO_R_NULL_PARAMETER, __FUNCTION__, __LINE__);
        return -1;
    }

    rlen = abcdk_enigma_ssl_read(bio_p->enigma_ssl, buf, len);
    if (rlen < 0)
        BIO_set_retry_read(bio); /*设置重试标志，非常重要。*/

    return rlen;
}

static int _abcdk_enigma_BIO_write_cb(BIO *bio, const char *buf, int len)
{
    abcdk_enigma_BIO_t *bio_p = (abcdk_enigma_BIO_t *)_abcdk_enigma_BIO_get_data(bio);
    int slen = 0;

    if(!(bio_p != NULL && bio_p->magic == ABCDK_ENIGMA_BIO_MAGIC))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_WRITE, BIO_R_BROKEN_PIPE, __FUNCTION__, __LINE__);
        return -1;
    }

    if (!(buf != NULL && len > 0))
    {
        ERR_put_error(ERR_LIB_BIO, BIO_F_BIO_WRITE, BIO_R_NULL_PARAMETER, __FUNCTION__, __LINE__);
        return -1;
    }


    slen = abcdk_enigma_ssl_write(bio_p->enigma_ssl, buf, len);
    if (slen < 0)
        BIO_set_retry_write(bio); /*设置重试标志，非常重要。*/

    return slen;
}

static long _abcdk_enigma_BIO_ctrl_cb(BIO *bio, int cmd, long num, void *ptr)
{
    abcdk_enigma_BIO_t *bio_p = (abcdk_enigma_BIO_t *)_abcdk_enigma_BIO_get_data(bio);
    int chk = 0;

    if(!(bio_p != NULL && bio_p->magic == ABCDK_ENIGMA_BIO_MAGIC))
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
            abcdk_enigma_ssl_set_fd(bio_p->enigma_ssl, fd, 0);
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
        ABCDK_PTR2I32(ptr, 0) = abcdk_enigma_ssl_get_fd(bio_p->enigma_ssl, 0);
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

static int _abcdk_enigma_BIO_create_cb(BIO *bio)
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

static int _abcdk_enigma_BIO_destroy_cb(BIO *bio)
{
    abcdk_enigma_BIO_t *bio_p = (abcdk_enigma_BIO_t *)_abcdk_enigma_BIO_get_data(bio);

    if(!bio_p)
        return 1;

    if(bio_p->magic != ABCDK_ENIGMA_BIO_MAGIC)
        return 0;

    abcdk_enigma_ssl_destroy(&bio_p->enigma_ssl);
    _abcdk_enigma_BIO_meth_free(bio_p->method);
    abcdk_heap_free(bio_p);

    return 1;
}


int abcdk_enigma_BIO_set_fd(BIO *bio, int fd)
{
    abcdk_enigma_BIO_t *bio_p;

    bio_p = _abcdk_enigma_BIO_get_data(bio);
    if(!bio_p || bio_p->magic != ABCDK_ENIGMA_BIO_MAGIC)
        return -1;

    abcdk_enigma_ssl_set_fd(bio_p->enigma_ssl,fd,0);

    return 0;
}

int abcdk_enigma_BIO_get_fd(BIO *bio)
{
    abcdk_enigma_BIO_t *bio_p;

    bio_p = _abcdk_enigma_BIO_get_data(bio);
    if(!bio_p || bio_p->magic != ABCDK_ENIGMA_BIO_MAGIC)
        return -1;

    return abcdk_enigma_ssl_get_fd(bio_p->enigma_ssl,0);
}

void abcdk_enigma_BIO_destroy(BIO **bio)
{
    BIO *bio_p;

    if(!bio ||!*bio)
        return;

    bio_p = *bio;
    *bio = NULL;

    BIO_free(bio_p);
}

BIO *abcdk_enigma_BIO_s_SSL(const char *file)
{
    abcdk_enigma_BIO_t *bio;
    BIO *openssl_bio;

    assert(file != NULL);
    
    bio = (abcdk_enigma_BIO_t*)abcdk_heap_alloc(sizeof(abcdk_enigma_BIO_t));
    if (!bio)
        goto ERR;

    bio->magic = ABCDK_ENIGMA_BIO_MAGIC;
    bio->enigma_ssl = abcdk_enigma_ssl_create_from_file(file);
    bio->method = _abcdk_enigma_BIO_meth_new(BIO_TYPE_SOURCE_SINK,"EnigmaSSL BIO");

    if (!bio->enigma_ssl || !bio->method)
        goto ERR;
    
    _abcdk_enigma_BIO_meth_set_write(bio->method,_abcdk_enigma_BIO_write_cb);
    _abcdk_enigma_BIO_meth_set_read(bio->method,_abcdk_enigma_BIO_read_cb);
    _abcdk_enigma_BIO_meth_set_ctrl(bio->method,_abcdk_enigma_BIO_ctrl_cb);
    _abcdk_enigma_BIO_meth_set_create(bio->method,_abcdk_enigma_BIO_create_cb);
    _abcdk_enigma_BIO_meth_set_destroy(bio->method,_abcdk_enigma_BIO_destroy_cb);

    openssl_bio = BIO_new(bio->method);
    if (!openssl_bio)
        goto ERR;

    /*关联到一起。*/
    _abcdk_enigma_BIO_set_data(openssl_bio, bio);

    /*关联成功后，清理野指针。*/
    bio = NULL;

    return openssl_bio;

ERR:

    if(openssl_bio)
        BIO_free(openssl_bio);

    if(bio)
    {
        abcdk_enigma_ssl_destroy(&bio->enigma_ssl);
        _abcdk_enigma_BIO_meth_free(bio->method);
        abcdk_heap_free(bio);
    }

    return NULL;
}

#endif // HEADER_BIO_H
