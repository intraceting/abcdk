/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/openssl/openssl.h"

/******************************************************************************************************/

#ifdef OPENSSL_VERSION_NUMBER
#if OPENSSL_VERSION_NUMBER < 0x10000000L

static pthread_mutex_t *_abcdk_openssl_crypto_locker_array = NULL;

/*锁回调函数。*/
static void _abcdk_openssl_crypto_locking_cb(int mode, int type, const char *file, int line)
{
    if (mode & CRYPTO_LOCK)
        pthread_mutex_lock(&(_abcdk_openssl_crypto_locker_array[type]));
    else
        pthread_mutex_unlock(&(_abcdk_openssl_crypto_locker_array[type]));
}

/*线程ID回调函数。*/ 
static unsigned long _abcdk_openssl_crypto_id_cb(void)
{
    return (unsigned long)pthread_self();
}

void _abcdk_openssl_crypto_setup()
{
    /*动态分配锁内存。*/
    _abcdk_openssl_crypto_locker_array = OPENSSL_malloc(CRYPTO_num_locks() * sizeof(pthread_mutex_t));

    /*初始化锁。*/
    for (int i = 0; i < CRYPTO_num_locks(); i++)
        pthread_mutex_init(&_abcdk_openssl_crypto_locker_array[i], NULL);

    /*设置锁和线程ID回调函数。*/
    CRYPTO_set_locking_callback(_abcdk_openssl_crypto_locking_cb);
    CRYPTO_set_id_callback(_abcdk_openssl_crypto_id_cb);
}

void _abcdk_openssl_crypto_cleanup()
{
    CRYPTO_set_locking_callback(NULL);

    for (int i = 0; i < CRYPTO_num_locks(); i++)
        pthread_mutex_destroy(&_abcdk_openssl_crypto_locker_array[i]);

    /*释放内存。*/
    OPENSSL_free(_abcdk_openssl_crypto_locker_array);
}

#endif //#if OPENSSL_VERSION_NUMBER < 0x10000000L
#endif //OPENSSL_VERSION_NUMBER

void abcdk_openssl_cleanup()
{
#ifdef OPENSSL_VERSION_NUMBER

    ERR_remove_thread_state(NULL);
    ERR_free_strings();

    OBJ_cleanup();
    RAND_cleanup();
    //ENGINE_cleanup();
    EVP_cleanup();
    CRYPTO_cleanup_all_ex_data();

    CONF_modules_free();
    CONF_modules_unload(1);
    SSL_COMP_free_compression_methods();

#if OPENSSL_VERSION_NUMBER < 0x10000000L
    _abcdk_openssl_crypto_cleanup();
#endif //#if OPENSSL_VERSION_NUMBER < 0x10000000L

#endif // OPENSSL_VERSION_NUMBER
}

void abcdk_openssl_init()
{
#ifdef OPENSSL_VERSION_NUMBER

    SSL_library_init();
    OpenSSL_add_all_algorithms();
    ERR_load_BIO_strings();
    ERR_load_crypto_strings();
    SSL_load_error_strings();

#if OPENSSL_VERSION_NUMBER < 0x10000000L
    _abcdk_openssl_crypto_setup();
#endif //#if OPENSSL_VERSION_NUMBER < 0x10000000L

#endif //OPENSSL_VERSION_NUMBER
}

/******************************************************************************************************/

#ifdef OPENSSL_VERSION_NUMBER

typedef struct _abcdk_openssl_pem_password_ctx
{
    const char *key;
    abcdk_object_t *passwd;
}abcdk_openssl_pem_password_ctx_t;

static int _abcdk_openssl_pem_password_cb(char *buf, int size, int rwflag, void *userdata)
{
    abcdk_openssl_pem_password_ctx_t *ctx = (abcdk_openssl_pem_password_ctx_t*)userdata;
    int chk;

    abcdk_object_unref(&ctx->passwd);//free.
    ctx->passwd = abcdk_getpass(NULL,"Enter passphrase for %s",ctx->key);

    chk = ABCDK_MIN((int)ctx->passwd->sizes[0],size);
    memcpy(buf,ctx->passwd->pptrs[0],chk);

    return chk;
}

void abcdk_openssl_bn_free(BIGNUM **bn)
{
    BIGNUM *bn_p;

    if(!bn || !*bn)
        return;

    bn_p = *bn;
    *bn = NULL;

    BN_clear_free(bn_p);
}

void abcdk_openssl_evp_pkey_free(EVP_PKEY **pkey)
{
    EVP_PKEY *pkey_p;

    if(!pkey || !*pkey)
        return;

    pkey_p = *pkey;
    *pkey = NULL;

    EVP_PKEY_free(pkey_p);
}

void abcdk_openssl_evp_cipher_ctx_free(EVP_CIPHER_CTX **cipher)
{
    EVP_CIPHER_CTX *cipher_p;

    if(!cipher || !*cipher)
        return;

    cipher_p = *cipher;
    *cipher = NULL;

    EVP_CIPHER_CTX_cleanup(cipher_p);
    EVP_CIPHER_CTX_free(cipher_p);
}

EVP_PKEY *abcdk_openssl_evp_pkey_load(const char *key, int pubkey, abcdk_object_t **passwd)
{
    EVP_PKEY *pkey = NULL;
    FILE *fp = NULL;
    abcdk_openssl_pem_password_ctx_t ctx = {0};

    assert(key != NULL);

    ctx.key = key;
    ctx.passwd = NULL;

    fp = fopen(key, "r");
    if (fp)
    {
        if (!pubkey)
            pkey = PEM_read_PrivateKey(fp, NULL, _abcdk_openssl_pem_password_cb, &ctx);
        else
            pkey = PEM_read_PUBKEY(fp, NULL, _abcdk_openssl_pem_password_cb, &ctx);

        fclose(fp);
    }
    
    if (passwd && ctx.passwd)
        *passwd = abcdk_object_refer(ctx.passwd);

    abcdk_object_unref(&ctx.passwd); // free.
    return pkey;
}

#endif //OPENSSL_VERSION_NUMBER

/******************************************************************************************************/

#ifdef HEADER_RSA_H

void abcdk_openssl_rsa_free(RSA **rsa)
{
    RSA *rsa_p;

    if(!rsa || !*rsa)
        return;

    rsa_p = *rsa;
    *rsa = NULL;

    RSA_free(rsa_p);
}

int abcdk_openssl_rsa_is_prikey(RSA *rsa) 
{
#if 0
    RSA *rsa_p;
    int chk;

    assert(rsa != NULL);

    rsa_p = RSAPrivateKey_dup(rsa);
    chk = (rsa_p != NULL);

    abcdk_openssl_rsa_free(&rsa_p);

    return chk;
#else //上面的代码，当传入的是公钥时，会有内存泄漏问题。

    const BIGNUM *d_p = NULL;

    assert(rsa != NULL);

#if OPENSSL_VERSION_NUMBER >= 0x10100000L
    RSA_get0_key(rsa, NULL, NULL, &d_p);
#else //#if OPENSSL_VERSION_NUMBER >= 0x10100000L
    d_p = rsa->d;
#endif //#if OPENSSL_VERSION_NUMBER >= 0x10100000L

    /*仅私钥中存在这个组件，公钥中没有。*/
    return (d_p != NULL);

#endif //
}

RSA *abcdk_openssl_rsa_create(int bits, unsigned long e)
{
#ifdef HEADER_BIO_H

    RSA *key = NULL;
    BIGNUM *bne = NULL;
    int chk;
    
    bne = BN_new();
    if(!bne)
        return NULL;

    chk = BN_set_word(bne,e);
    if(chk <= 0)
        goto ERR;

    key = RSA_new();
    if(!key)
        goto ERR;

    chk = RSA_generate_key_ex(key, bits, bne, NULL);
    if (chk <= 0)
        goto ERR;

    abcdk_openssl_bn_free(&bne);
    return key;

ERR:
        
    abcdk_openssl_rsa_free(&key);
    abcdk_openssl_bn_free(&bne);

#endif //HEADER_BIO_H

    return NULL;
}

RSA *abcdk_openssl_rsa_load(const char *key, int pubkey, abcdk_object_t **passwd)
{
    EVP_PKEY *pkey;
    RSA *rsa_ctx;

    assert(key != NULL);

    pkey = abcdk_openssl_evp_pkey_load(key, pubkey, NULL);
    if (!pkey)
        return NULL;

    rsa_ctx = EVP_PKEY_get1_RSA(pkey);
    abcdk_openssl_evp_pkey_free(&pkey);

    if(!rsa_ctx)
        return NULL;

    return rsa_ctx;
}

abcdk_object_t *abcdk_openssl_rsa_export(RSA *rsa)
{
#ifdef HEADER_BIO_H

    abcdk_object_t *out;
    BIO *pem_bio = NULL;
    char *key_p = NULL;
    long key_l = 0;
    int chk;

    assert(rsa != NULL);
    
    pem_bio = BIO_new(BIO_s_mem());
    if(!pem_bio)
        return NULL;

    if(abcdk_openssl_rsa_is_prikey(rsa))
        chk = PEM_write_bio_RSAPrivateKey(pem_bio, rsa, NULL, NULL, 0, NULL, NULL);
    else 
        chk = PEM_write_bio_RSA_PUBKEY(pem_bio, rsa);

    if(chk != 1)
        goto ERR;

    key_l = BIO_get_mem_data(pem_bio, &key_p);
    if(key_l <=0 || key_p == NULL)
        goto ERR;

    out = abcdk_object_copyfrom(key_p,key_l);
    BIO_free_all(pem_bio);

    return out;

ERR:

    if(pem_bio)
        BIO_free_all(pem_bio);

#endif //HEADER_BIO_H

    return NULL;
}

static int _abcdk_openssl_rsa_update_fragment(uint8_t *out, int out_max, const uint8_t *in, int in_len,
                                              int enc, RSA *rsa, int pubkey, int pt_bsize, int ct_bsize, int padding)
{
    uint8_t tmpbuf[1024];
    int chk;

    /*
     * |DATA    |PADDING  |
     * |N bytes |11 bytes |
     */

    if (enc)
    {
        assert(in_len <= pt_bsize);
        assert(out_max >= ct_bsize);

        memcpy(tmpbuf, in, in_len);
        RAND_bytes(tmpbuf + in_len, pt_bsize - in_len);

        if (!pubkey)
            chk = RSA_private_encrypt(pt_bsize, tmpbuf, out, rsa, padding);
        else 
            chk = RSA_public_encrypt(pt_bsize, tmpbuf, out, rsa, padding);

        if (chk <= 0)
            return -3;

        return ct_bsize;
    }
    else
    {
        assert(in_len == ct_bsize);
        assert(out_max >= pt_bsize);

        if (!pubkey)
            chk = RSA_private_decrypt(in_len, in, tmpbuf, rsa, padding);
        else 
            chk = RSA_public_decrypt(in_len, in, tmpbuf, rsa, padding);

        if (chk <= 0)
            return -3;

        memcpy(out, tmpbuf, pt_bsize);

        return pt_bsize;
    }

    return -1;
}

abcdk_object_t *abcdk_openssl_rsa_update(RSA *rsa, const uint8_t *in, int in_len, int enc)
{
    abcdk_object_t *out;
    int pubkey,blocks,pt_bsize, ct_bsize;
    int chk;

    assert(rsa != NULL && in != NULL && in_len > 0);

    /*判断KEY类型。*/
    pubkey = !abcdk_openssl_rsa_is_prikey(rsa);

    /*计算明文和密文的块长度。*/
    pt_bsize = RSA_size(rsa) - RSA_PKCS1_PADDING_SIZE;
    ct_bsize = RSA_size(rsa);

    if (enc)
    {
        blocks = abcdk_align(in_len, pt_bsize) / pt_bsize;

        out = abcdk_object_alloc2(blocks * ct_bsize);
        if (!out)
            return NULL;

        for (int i = 0; i < blocks; i++)
        {
            if (i < (blocks - 1) || (in_len % pt_bsize) == 0)
            {
                _abcdk_openssl_rsa_update_fragment(out->pptrs[0] + i * ct_bsize, ct_bsize, in + i * pt_bsize, pt_bsize,
                                                   1, rsa, pubkey, pt_bsize, ct_bsize, RSA_PKCS1_PADDING);
            }
            else
            {
                /*明文没有块对齐时，最后一块需要单独计算。*/
                _abcdk_openssl_rsa_update_fragment(out->pptrs[0] + i * ct_bsize, ct_bsize, in + i * pt_bsize, in_len % pt_bsize,
                                                   1, rsa, pubkey, pt_bsize, ct_bsize, RSA_PKCS1_PADDING);
            }
        }
    }
    else
    {
        /*密文必须是块对齐的。*/
        if ((in_len % ct_bsize) != 0)
            return NULL;

        blocks = in_len / ct_bsize;

        out = abcdk_object_alloc2(blocks * pt_bsize);
        if (!out)
            return NULL;

        for (int i = 0; i < blocks; i++)
        {
            _abcdk_openssl_rsa_update_fragment(out->pptrs[0] + i * pt_bsize, pt_bsize, in + i * ct_bsize, ct_bsize,
                                               0, rsa, pubkey, pt_bsize, ct_bsize, RSA_PKCS1_PADDING);
        }
    }

    return out;
}

#endif //HEADER_RSA_H

/******************************************************************************************************/

#ifdef HEADER_HMAC_H

void abcdk_openssl_hmac_free(HMAC_CTX **ctx)
{
    HMAC_CTX *ctx_p;

    if(!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
    HMAC_CTX_cleanup(ctx_p);//不可以省略。
    abcdk_heap_free(ctx_p);
#else //#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
    HMAC_CTX_free(ctx_p);
#endif //#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
}

/**申请。*/
HMAC_CTX *abcdk_openssl_hmac_alloc()
{
    HMAC_CTX *ctx;

#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
    ctx = (HMAC_CTX*)abcdk_heap_alloc(sizeof(HMAC_CTX));
#else //#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
    ctx = HMAC_CTX_new();
#endif //#if OPENSSL_VERSION_NUMBER <= 0x100020bfL

    if(!ctx)
        return NULL;
    
#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
    HMAC_CTX_init(ctx);//不可以省略。
#endif //#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
    
    return ctx;
}

int abcdk_openssl_hmac_init(HMAC_CTX *hmac, const void *key, int len, int type)
{
    int chk = -1;

    assert(hmac != NULL && key != NULL && len > 0);
    assert(type >= ABCDK_OPENSSL_HMAC_MD2 && type <= ABCDK_OPENSSL_HMAC_WHIRLPOOL);

    if (0)
        assert(0);
#ifndef OPENSSL_NO_MD2
    else if (type == ABCDK_OPENSSL_HMAC_MD2)
        chk = HMAC_Init_ex(hmac, key, len, EVP_md2(), NULL);
#endif
#ifndef OPENSSL_NO_MD4
    else if (type == ABCDK_OPENSSL_HMAC_MD4)
        chk = HMAC_Init_ex(hmac, key, len, EVP_md4(), NULL);
#endif
#ifndef OPENSSL_NO_MD5
    else if (type == ABCDK_OPENSSL_HMAC_MD5)
        chk = HMAC_Init_ex(hmac, key, len, EVP_md5(), NULL);
#endif
#ifndef OPENSSL_NO_SHA
#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
    else if (type == ABCDK_OPENSSL_HMAC_SHA)
        chk = HMAC_Init_ex(hmac, key, len, EVP_sha(), NULL);
#endif
    else if (type == ABCDK_OPENSSL_HMAC_SHA1)
        chk = HMAC_Init_ex(hmac, key, len, EVP_sha1(), NULL);
#endif
#ifndef OPENSSL_NO_SHA256
    else if (type == ABCDK_OPENSSL_HMAC_SHA224)
        chk = HMAC_Init_ex(hmac, key, len, EVP_sha224(), NULL);
    else if (type == ABCDK_OPENSSL_HMAC_SHA256)
        chk = HMAC_Init_ex(hmac, key, len, EVP_sha256(), NULL);
#endif
#ifndef OPENSSL_NO_SHA512
    else if (type == ABCDK_OPENSSL_HMAC_SHA384)
        chk = HMAC_Init_ex(hmac, key, len, EVP_sha384(), NULL);
    else if (type == ABCDK_OPENSSL_HMAC_SHA512)
        chk = HMAC_Init_ex(hmac, key, len, EVP_sha512(), NULL);
#endif
#ifndef OPENSSL_NO_RIPEMD
    else if (type == ABCDK_OPENSSL_HMAC_RIPEMD160)
        chk = HMAC_Init_ex(hmac, key, len, EVP_ripemd160(), NULL);
#endif
#ifndef OPENSSL_NO_WHIRLPOOL
    else if (type == ABCDK_OPENSSL_HMAC_WHIRLPOOL)
        chk = HMAC_Init_ex(hmac, key, len, EVP_whirlpool(), NULL);
#endif

    return (chk == 1) ? 0 : -1;
}

#endif //HEADER_HMAC_H

/******************************************************************************************************/

#ifdef HEADER_X509_H

void abcdk_openssl_x509_free(X509 **x509)
{
    X509 *x509_p;

    if(!x509 || !*x509)
        return;

    x509_p = *x509;
    *x509 = NULL;

    X509_free(x509_p);
}

abcdk_object_t *abcdk_openssl_cert_dump(X509 *x509)
{
# ifndef OPENSSL_NO_BIO
    BIO *mem;
    char *data_p = NULL;
    long data_l = 0;
    abcdk_object_t *cert_info;

    assert(x509 != NULL);

    mem = BIO_new(BIO_s_mem());
    if(!mem)
        return NULL;

    X509_print(mem, x509);

    data_l = BIO_get_mem_data(mem, &data_p);
    cert_info = abcdk_object_copyfrom(data_p,data_l);        
    BIO_free(mem);

    return cert_info;
#else //# ifndef OPENSSL_NO_BIO
    return NULL;
#endif  //# ifndef OPENSSL_NO_BIO
}

abcdk_object_t *abcdk_openssl_cert_verify_error_dump(X509_STORE_CTX *store_ctx)
{
    int err_num,err_depth;
    X509 *err_cert = NULL;
    abcdk_object_t *err_cert_info = NULL;
    abcdk_object_t *err_info = NULL;

    assert(store_ctx != NULL);

    err_info = abcdk_object_alloc2(1024*1024);
    if(!err_info)
        return NULL;

    err_num = X509_STORE_CTX_get_error(store_ctx);
    err_depth = X509_STORE_CTX_get_error_depth(store_ctx);

    err_cert = X509_STORE_CTX_get_current_cert(store_ctx);
    if(err_cert)
        err_cert_info = abcdk_openssl_cert_dump(err_cert);

    err_info = abcdk_object_printf(1024 * 1024, "Error depth: %d\nError code: %d\nError message: %s\nError certificate: {\n%s\n}\n",
                                   err_depth, err_num, X509_verify_cert_error_string(err_num), (err_cert_info ? err_cert_info->pstrs[0] : ""));

    return err_info;
}

RSA *abcdk_openssl_cert_get_rsa_pubkey(X509 *x509)
{
    RSA *rsa = NULL;
    EVP_PKEY *pkey = NULL;

    assert(x509 != NULL);

    pkey = X509_get_pubkey(x509);
    if(!pkey)
        return NULL;

#ifndef OPENSSL_NO_RSA
    rsa = EVP_PKEY_get1_RSA(pkey);
#endif //OPENSSL_NO_RSA

    abcdk_openssl_evp_pkey_free(&pkey);

    return rsa;
}

abcdk_object_t *abcdk_openssl_cert_to_pem(X509 *leaf_cert,STACK_OF(X509) *cert_chain)
{
# ifndef OPENSSL_NO_BIO
    BIO *bio;
    BUF_MEM *bmem_p;
    X509 *cert_p; 
    abcdk_object_t *pem_info;

    assert(leaf_cert!= NULL || cert_chain != NULL);

    bio = BIO_new(BIO_s_mem());
    if (bio == NULL)
        return NULL;

    if (PEM_write_bio_X509(bio, leaf_cert) != 1)
        goto ERR;

    for (int i = 0; i < sk_X509_num(cert_chain); i++) 
    {
        cert_p = sk_X509_value(cert_chain, i);
        if (PEM_write_bio_X509(bio, cert_p) != 1) 
            goto ERR;
    }

    BIO_get_mem_ptr(bio, &bmem_p);
    BIO_set_close(bio, BIO_NOCLOSE); // 保留内存 BIO 的数据
    BIO_free(bio);

    pem_info = abcdk_object_copyfrom(bmem_p->data, bmem_p->length);
    BUF_MEM_free(bmem_p);

    return pem_info;

ERR:

    BIO_free(bio);
    return NULL;
#else //# ifndef OPENSSL_NO_BIO
    return NULL;
#endif  //# ifndef OPENSSL_NO_BIO
}

X509_CRL *abcdk_openssl_crl_load(const char *crl)
{
    X509_CRL *ctx = NULL;
    FILE *fp = NULL;

    assert(crl != NULL);

    fp = fopen(crl, "r");
    if(!fp)
        return NULL;
    
    ctx = PEM_read_X509_CRL(fp, NULL, NULL, NULL);
        
    fclose(fp);

    return ctx;
}

X509 *abcdk_openssl_cert_load(const char *crt)
{
    X509 *ctx = NULL;
    FILE *fp = NULL;

    assert(crt != NULL);

    fp = fopen(crt, "r");
    if(!fp)
        return NULL;
    
    ctx = PEM_read_X509(fp, NULL, NULL, NULL);
    fclose(fp);

    return ctx;
}


X509 *abcdk_openssl_cert_father_find(X509 *leaf_cert,const char *ca_path,const char *pattern)
{
    X509_NAME *issuer_name,*subject_name;
    X509 *father_cert = NULL;
    abcdk_tree_t *dir = NULL;
    char cert_file[PATH_MAX];
    int chk;

    assert(leaf_cert != NULL && ca_path != NULL);

    chk = abcdk_dirent_open(&dir,ca_path);
    if(chk != 0)
        return NULL;

    while(1)
    {
        memset(cert_file,0,PATH_MAX);
        chk = abcdk_dirent_read(dir,pattern,cert_file,1);
        if(chk != 0)
            break;

        father_cert = abcdk_openssl_cert_load(cert_file);
        if(!father_cert)
            continue;

        issuer_name = X509_get_issuer_name(leaf_cert);
        subject_name = X509_get_subject_name(father_cert);

        /*
         * 1：自签名证书的颁发者是自己，排除。
         * 2：如果当前证书的主题名和叶证书的颁发者不同，排除。
         */
        if (X509_cmp(leaf_cert, father_cert) == 0 || X509_NAME_cmp(issuer_name, subject_name) != 0)
        {
            X509_free(father_cert);
            father_cert = NULL;
            continue;
        }
        else
        {
            break;
        }
    }

    abcdk_tree_free(&dir);

    return father_cert;
}

STACK_OF(X509) *abcdk_openssl_cert_chain_load(X509 *leaf_cert, const char *ca_path,const char *pattern)
{
    STACK_OF(X509) *cert_chain;
    X509 *curt_cert,*father_cert;

    assert(leaf_cert != NULL && ca_path != NULL);

    cert_chain = sk_X509_new_null();
    if(!cert_chain)
        return NULL;

    /*从叶证书开始遍历。*/
    curt_cert = leaf_cert;
 
    while (1)
    {
        father_cert = abcdk_openssl_cert_father_find(curt_cert, ca_path, pattern);
        if (father_cert)
        {
            sk_X509_push(cert_chain, father_cert);
            curt_cert = father_cert;
            father_cert = NULL;
        }
        else
        {
            curt_cert = NULL;
            break;
        }

    }

    return cert_chain;

ERR:

    if(cert_chain)
        sk_X509_pop_free(cert_chain, X509_free);

    return NULL;
}

STACK_OF(X509) *abcdk_openssl_cert_chain_load_mem(const char *buf,int len)
{
    STACK_OF(X509) *cert_chain;
    X509 *cert_p = NULL;
    FILE *fp = NULL;

    assert(buf != NULL);

    /*可能需要自动计算。*/
    if(len <= 0)
        len = strlen(buf);

    if(len <= 0)
        return NULL;
    
    cert_chain = sk_X509_new_null();
     if(!cert_chain)
        return NULL;

    fp = fmemopen((void*)buf,len, "r");
    if (!fp) 
        goto ERR;

    /*循环读取所以证书。*/
    while (cert_p = PEM_read_X509(fp, NULL, NULL, NULL)) 
        sk_X509_push(cert_chain, cert_p);
    
    fclose(fp);

    return cert_chain;

ERR:

    if(fp)
        fclose(fp);
    if(cert_chain)
        sk_X509_pop_free(cert_chain, X509_free);
    
    return NULL;
}

X509_STORE *abcdk_openssl_cert_load_locations(const char *ca_file, const char *ca_path)
{
    X509_STORE *store;
    int chk;

    store = X509_STORE_new();
    if(!store)
        return NULL;

    if(!ca_file && !ca_path)
        return store;

    chk = X509_STORE_load_locations(store,ca_file,ca_path);
    if(chk == 1)
        return store;

ERR:

    X509_STORE_free(store);
    return NULL;
}

X509_STORE_CTX *abcdk_openssl_cert_verify_prepare(X509_STORE *store,X509 *leaf_cert,STACK_OF(X509) *cert_chain)
{
    X509_STORE_CTX *store_ctx = NULL;
    int chk;

    assert(store != NULL && (leaf_cert != NULL && cert_chain != NULL)  || leaf_cert != NULL);

    store_ctx = X509_STORE_CTX_new();
    if (!store_ctx)
        return NULL;

    chk = X509_STORE_CTX_init(store_ctx, store, leaf_cert, cert_chain);
    if (chk == 1)
        return store_ctx;

ERR:

    if(store_ctx)
    {
        X509_STORE_CTX_cleanup(store_ctx);
        X509_STORE_CTX_free(store_ctx);
    }

    return NULL;
}


#endif //HEADER_X509_H

/******************************************************************************************************/

#ifdef HEADER_SSL_H

void abcdk_openssl_ssl_ctx_free(SSL_CTX **ctx)
{
    SSL_CTX *ctx_p = NULL;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    SSL_CTX_free(ctx_p);
}

SSL_CTX *abcdk_openssl_ssl_ctx_alloc(int server,const char *cafile,const char *capath, int chk_crl,X509 *use_crt, EVP_PKEY *use_key)
{
    const SSL_METHOD *method = NULL;
    SSL_CTX *ctx = NULL;
    int chk;

#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
    method = (server ? TLSv1_2_server_method() : TLSv1_2_client_method());
#else
    method = (server ? TLS_server_method() : TLS_client_method());
#endif

    ctx = SSL_CTX_new(method);
    if(!ctx)
        return NULL;

#if OPENSSL_VERSION_NUMBER >= 0x1010100F
#ifdef TLS1_2_VERSION
    SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
#endif //TLS1_2_VERSION

#ifdef TLS_MAX_VERSION
    SSL_CTX_set_max_proto_version(ctx, TLS_MAX_VERSION);
#endif //TLS_MAX_VERSION
#endif //OPENSSL_VERSION_NUMBER >= 0x1010100F

    if (cafile || capath)
    {
        chk = SSL_CTX_load_verify_locations(ctx, cafile, capath);
        if (chk != 1)
        {
            if (cafile)
                abcdk_trace_printf(LOG_WARNING, "加载CA证书('%s')错误。\n", cafile);
            if (capath)
                abcdk_trace_printf(LOG_WARNING, "加载CA路径('%s')错误。\n", capath);

            goto ERR;
        }
    }

#if OPENSSL_VERSION_NUMBER >= 0x10002000L

    X509_VERIFY_PARAM *param = SSL_CTX_get0_param(ctx);
    if(!param && chk_crl)
        goto ERR;

    chk = X509_VERIFY_PARAM_set_purpose(param, X509_PURPOSE_ANY);
    if(chk != 1)
        goto ERR;

    if(chk_crl == 2)
        chk = X509_VERIFY_PARAM_set_flags(param, X509_V_FLAG_CRL_CHECK | X509_V_FLAG_CRL_CHECK_ALL);
    else if(chk_crl == 1)
        chk = X509_VERIFY_PARAM_set_flags(param, X509_V_FLAG_CRL_CHECK);
    else 
        chk = 1;

    if(chk != 1)
        goto ERR;

#endif //OPENSSL_VERSION_NUMBER >= 0x10002000L

    /*禁止会话复用。*/
    SSL_CTX_set_session_cache_mode(ctx, SSL_SESS_CACHE_OFF);
    
#ifdef SSL_OP_NO_TICKET
    /*禁用会话票据*/
    SSL_CTX_set_options(ctx, SSL_OP_NO_TICKET);
#endif //SSL_OP_NO_TICKET

    if (use_crt)
    {
        chk = SSL_CTX_use_certificate(ctx, use_crt);
        if (chk != 1)
            goto ERR;
    }

    if (use_key)
    {
        chk = SSL_CTX_use_PrivateKey(ctx, use_key);
        if (chk != 1)
            goto ERR;
    }

    if (use_crt && use_key)
    {
        chk = SSL_CTX_check_private_key(ctx);
        if (chk != 1)
            goto ERR;
    }

    if (cafile || capath)
        SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT, NULL);
    else 
        SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, NULL);

    return ctx;

ERR:

    abcdk_openssl_ssl_ctx_free(&ctx);
    return NULL;
}

#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
static int _abcdk_openssl_alpn_select_cb(SSL *ssl, const unsigned char **out, unsigned char *outlen,
                                         const unsigned char *in, unsigned int inlen, void *arg)
{
    const uint8_t *proto_p;
    const unsigned char *srv;
    unsigned int srvlen;

    proto_p = (const uint8_t *)arg;

    if (!proto_p)
        return SSL_TLSEXT_ERR_ALERT_FATAL;

    srv = proto_p;
    srvlen = strlen(proto_p);

    /*服务端在客户端支持的协议列表中选择一个支持协议，从左到右按顺序匹配。*/
#ifndef OPENSSL_NO_TLSEXT
    if (SSL_select_next_proto((unsigned char **)out, outlen, in, inlen, srv, srvlen) != OPENSSL_NPN_NEGOTIATED)
#endif //OPENSSL_NO_TLSEXT
    {
        return SSL_TLSEXT_ERR_ALERT_FATAL;
    }

    return SSL_TLSEXT_ERR_OK;
}
#endif // TLSEXT_TYPE_application_layer_protocol_negotiation

int abcdk_openssl_ssl_ctx_set_alpn(SSL_CTX *ctx,const uint8_t *next_proto,const char *cipher_list)
{
    int chk;

    assert(ctx != NULL && next_proto != NULL);

#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
    SSL_CTX_set_alpn_select_cb(ctx, _abcdk_openssl_alpn_select_cb, (void *)next_proto);
#endif // TLSEXT_TYPE_application_layer_protocol_negotiation

    if(cipher_list)
    {
        chk = SSL_CTX_set_cipher_list(ctx, cipher_list);
        if(chk != 1)
            return -1;
    }

    return 0;
}

void abcdk_openssl_ssl_free(SSL **ssl)
{
    SSL *ssl_p;
    int chk,ssl_chk,ssl_err;
    int fd;

    if (!ssl || !*ssl)
        return;

    ssl_p = *ssl;
    *ssl = NULL;

    fd = SSL_get_fd(ssl_p);
    if (fd < 0)
        goto final;

    // while (1)
    // {
    //     ssl_chk = SSL_shutdown(ssl_p);
    //     if (ssl_chk == 1)
    //         break;
    //     else if (ssl_chk == 0)
    //         continue;

    //     ssl_err = SSL_get_error(ssl_p, ssl_chk);
    //     if (ssl_err == SSL_ERROR_WANT_WRITE)
    //         abcdk_poll(fd, 0x02, 10);
    //     else if (ssl_err == SSL_ERROR_WANT_READ)
    //         abcdk_poll(fd, 0x01, 10);
    //     else
    //         break;
    // }

final:

    SSL_free(ssl_p);
}

SSL *abcdk_openssl_ssl_alloc(SSL_CTX *ctx)
{
    SSL *ssl;
    assert(ctx != NULL);

    ssl = SSL_new(ctx);
    if(!ssl)
        return NULL;
    
#ifdef SSL_OP_NO_RENEGOTIATION
    /*禁止重新协商。*/
    SSL_set_options(ssl, SSL_OP_NO_RENEGOTIATION);
#endif //SSL_OP_NO_RENEGOTIATION

    return ssl;
}

int abcdk_openssl_ssl_handshake(int fd, SSL *ssl, int server, time_t timeout)
{
    int err;
    int chk;

    assert(fd >= 0 && ssl != NULL);

    if (SSL_get_fd(ssl) != fd)
    {
        chk = SSL_set_fd(ssl, fd);
        if (chk != 1)
            return -1;

        if (server)
            SSL_set_accept_state(ssl);
        else
            SSL_set_connect_state(ssl);
    }

try_again:

    chk = SSL_do_handshake(ssl);
    if (chk == 1)
        return 0;

    err = SSL_get_error(ssl, chk);
    if (err == SSL_ERROR_WANT_WRITE)
    {
        chk = abcdk_poll(fd, 0x02, timeout);
        if (chk <= 0)
            return -2;

        goto try_again;
    }
    else if (err == SSL_ERROR_WANT_READ)
    {
        chk = abcdk_poll(fd, 0x01, timeout);
        if (chk <= 0)
            return -2;

        goto try_again;
    }
    else
    {
       return -3;
    }

    return 0;
}

int abcdk_openssl_ssl_get_alpn_selected(SSL *ssl,char buf[256])
{
    const uint8_t *ver_p;
    unsigned int ver_l;

#ifdef TLSEXT_TYPE_application_layer_protocol_negotiation
    SSL_get0_alpn_selected(ssl, (const uint8_t **)&ver_p, &ver_l);
    if (ver_p != NULL && ver_l > 0)
    {
        memcpy(buf,ver_p,ABCDK_MIN(255,ver_l));
    }

    return 0;
#else 
    return -1;
#endif // TLSEXT_TYPE_application_layer_protocol_negotiation
}

#endif //HEADER_SSL_H

/******************************************************************************************************/
