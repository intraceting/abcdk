/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#include "abcdk/ssl/openssl.h"

/******************************************************************************************************/

#ifdef HEADER_AES_H

size_t abcdk_openssl_aes_set_key(AES_KEY *key, const void *pwd, size_t len, uint8_t padding, int encrypt)
{
    uint8_t key_buf[32] = {0};
    size_t key_bits = 0;
    int chk;

    assert(key != NULL && pwd != NULL && len > 0);

    if (len <= 16)
    {
        memset(key_buf, padding, 16);
        memcpy(key_buf, pwd, len);
        key_bits = 128;
    }
    else if (len <= 24)
    {
        memset(key_buf, padding, 24);
        memcpy(key_buf, pwd, len);
        key_bits = 192;
    }
    else if (len <= 32)
    {
        memset(key_buf, padding, 32);
        memcpy(key_buf, pwd, len);
        key_bits = 256;
    }

    if (key_bits)
    {
        if (encrypt)
            chk = AES_set_encrypt_key(key_buf, key_bits, key);
        else
            chk = AES_set_decrypt_key(key_buf, key_bits, key);

        assert(chk == 0);
    }

    return key_bits;
}

size_t abcdk_openssl_aes_set_iv(uint8_t *iv, const void *salt, size_t len, uint8_t padding)
{
    size_t iv_bytes = 0;

    assert(iv != NULL && salt != NULL && len > 0);

    if (len <= AES_BLOCK_SIZE)
    {
        memset(iv, padding, AES_BLOCK_SIZE);
        memcpy(iv, salt, len);
        iv_bytes = AES_BLOCK_SIZE;
    }
    else if (len <= AES_BLOCK_SIZE * 2)
    {
        memset(iv, padding, AES_BLOCK_SIZE * 2);
        memcpy(iv, salt, len);
        iv_bytes = AES_BLOCK_SIZE * 2;
    }
    else if (len <= AES_BLOCK_SIZE * 4)
    {
        memset(iv, padding, AES_BLOCK_SIZE * 4);
        memcpy(iv, salt, len);
        iv_bytes = AES_BLOCK_SIZE * 4;
    }

    return iv_bytes;
}

#endif //HEADER_AES_H

/******************************************************************************************************/

#ifdef HEADER_RSA_H

int abcdk_openssl_rsa_padding_size(int padding)
{
    int size = -1;

    switch (padding)
    {
    case RSA_PKCS1_PADDING:
        size = RSA_PKCS1_PADDING_SIZE; //=11
        break;
    case RSA_PKCS1_OAEP_PADDING:
        size = 42;
        break;
    case RSA_NO_PADDING:
    default:
        size = 0;
        break;
    }

    return size;
}

RSA *abcdk_openssl_rsa_create(int bits, unsigned long e)
{
    RSA *key = NULL;
    BIGNUM *bne = NULL;
    int chk;
    
    bne = BN_new();
    if(!bne)
        ABCDK_ERRNO_AND_GOTO1(ENOMEM,final);

    chk = BN_set_word(bne,e);
    if(chk <= 0)
        ABCDK_ERRNO_AND_GOTO1(EINVAL,final);

    key = RSA_new();
    if(!key)
        ABCDK_ERRNO_AND_GOTO1(ENOMEM,final);

    chk = RSA_generate_key_ex(key, bits, bne, NULL);
    if (chk <= 0)
    {
        RSA_free(key);
        key = NULL;
    }

final:
        
    if(bne)
        BN_clear_free(bne);

    return key;
}

RSA *abcdk_openssl_rsa_from_fp(FILE *fp, int type, const char *pwd)
{
    RSA *key = NULL;

    assert(fp != NULL);

    if (type)
        key = PEM_read_RSAPrivateKey(fp, NULL, NULL, (void *)pwd);
    else
        key = PEM_read_RSAPublicKey(fp, NULL, NULL, (void *)pwd);

    return key;
}

RSA *abcdk_openssl_rsa_from_file(const char *file, int type, const char *pwd)
{
    FILE *fp;
    RSA *key = NULL;

    assert(file != NULL);

    fp = fopen(file, "r");
    if (!fp)
        ABCDK_ERRNO_AND_RETURN1(errno, NULL);

    key = abcdk_openssl_rsa_from_fp(fp, type, pwd);

    /*不要忘记关闭。*/
    fclose(fp);

    return key;
}

int abcdk_openssl_rsa_to_fp(FILE *fp, RSA *key, int type, const char *pwd)
{
    int chk;

    assert(fp != NULL && key != NULL);

    if (type)
        chk = PEM_write_RSAPrivateKey(fp, key, NULL, NULL, 0, NULL, NULL);
    else
        chk = PEM_write_RSAPublicKey(fp, key);

    return chk;
}

int abcdk_openssl_rsa_to_file(const char *file, RSA *key, int type, const char *pwd)
{
    FILE *fp;
    int chk;

    assert(file != NULL && key != NULL);

    fp = fopen(file, "w");
    if (!fp)
        ABCDK_ERRNO_AND_RETURN1(errno, -1);

    chk = abcdk_openssl_rsa_to_fp(fp, key, type, pwd);

    /*不要忘记关闭。*/
    fclose(fp);

    return chk;
}

int abcdk_openssl_rsa_size(RSA *key)
{
    assert(key != NULL);

    return RSA_size(key);
}

int abcdk_openssl_rsa_encrypt(void *dst, const void *src, int len, RSA *key, int type, int padding)
{
    int chk;
    int ksize;
    int psize;

    assert(dst != NULL && src != NULL && len > 0 && key != NULL);

    /*
     * --------------------
     * | Buffer[KEY_SIZE] |
     * --------------------
     * | 数据       | 盐   |
     * --------------------
    */
    ksize = abcdk_openssl_rsa_size(key);
    psize = abcdk_openssl_rsa_padding_size(padding);
    assert((ksize - psize) == len);

    if (type)
        chk = RSA_private_encrypt(len, (uint8_t *)src, (uint8_t *)dst, key, padding);
    else
        chk = RSA_public_encrypt(len, (uint8_t *)src, (uint8_t *)dst, key, padding);

    return chk;
}

int abcdk_openssl_rsa_decrypt(void *dst, const void *src, int len, RSA *key, int type, int padding)
{
    int chk;
    int ksize;

    assert(dst != NULL && src != NULL && len > 0 && key != NULL);

    /*
     * --------------------
     * | Buffer[KEY_SIZE] |
     * --------------------
     * | 数据              |
     * --------------------
    */
    ksize = abcdk_openssl_rsa_size(key);
    assert(ksize == len);

    if (type)
        chk = RSA_private_decrypt(len, (uint8_t *)src, (uint8_t *)dst, key, padding);
    else
        chk = RSA_public_decrypt(len, (uint8_t *)src, (uint8_t *)dst, key, padding);

    return chk;
}

#endif //EADER_RSA_H

/******************************************************************************************************/

#ifdef HEADER_HMAC_H

int abcdk_openssl_hmac_init(HMAC_CTX *hmac, const void *key, int len, int type)
{
    int chk = -1;

    assert(hmac != NULL && key != NULL && len > 0);
    assert(type >= ABCDK_OPENSSL_HMAC_MD2 && type <= ABCDK_OPENSSL_HMAC_WHIRLPOOL);

    /*不可以省略。*/
#if OPENSSL_VERSION_NUMBER <= 0x100020bfL
    HMAC_CTX_init(hmac);
#endif 

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

/******************************************************************************************************/

#ifdef HEADER_SSL_H

abcdk_object_t *abcdk_openssl_dump_crt(X509 *x509)
{
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
}

RSA *abcdk_openssl_pubkey_crt(X509 *x509)
{
    RSA *rsa = NULL;
    EVP_PKEY *pkey = NULL;

    assert(x509 != NULL);

    pkey = X509_get_pubkey(x509);
    if(!pkey)
        return NULL;
#ifndef OPENSSL_NO_RSA
    rsa = EVP_PKEY_get1_RSA(pkey);
#endif

    EVP_PKEY_free(pkey);

    return rsa;
}

X509 *abcdk_openssl_load_crt(const char *crt, const char *pwd)
{
    X509 *ctx = NULL;
    FILE *fp = NULL;

    assert(crt != NULL);

    fp = fopen(crt, "r");
    if(!fp)
        return NULL;
    
    ctx = PEM_read_X509(fp, NULL, NULL, (void*)pwd);
        
    fclose(fp);

    return ctx;
}

X509_CRL *abcdk_openssl_load_crl(const char *crl, const char *pwd)
{
    X509_CRL *ctx = NULL;
    FILE *fp = NULL;

    assert(crl != NULL);

    fp = fopen(crl, "r");
    if(!fp)
        return NULL;
    
    ctx = PEM_read_X509_CRL(fp, NULL, NULL, (void*)pwd);
        
    fclose(fp);

    return ctx;
}

int abcdk_openssl_load_crt2store(X509_STORE *store,...)
{
    X509 *crt = NULL;
    const char *file_p = NULL;
    int count = 0;

    assert (store != NULL);

    va_list vaptr;
    va_start(vaptr, store);

    for (;; count++)
    {
        file_p = va_arg(vaptr, const char *);
        if (!file_p)
            break;

        crt = abcdk_openssl_load_crt(file_p, NULL);
        if (!crt)
            break;

        if (X509_STORE_add_cert(store, crt) != 1)
            break;

        X509_free(crt);
        crt = NULL;
    }

    va_end(vaptr);

    /*很重要，因为在循环中有未释放的可能。*/
    if(crt)
        X509_free(crt);

    return count;
}

int abcdk_openssl_load_crl2store(X509_STORE *store,...)
{
    X509_CRL *crl = NULL;
    const char *file_p = NULL;
    int count = 0;

    assert (store != NULL);

    va_list vaptr;
    va_start(vaptr, store);

    for (;; count++)
    {
        file_p = va_arg(vaptr, const char *);
        if (!file_p)
            break;

        crl = abcdk_openssl_load_crl(file_p, NULL);
        if (!crl)
            break;

        if (X509_STORE_add_crl(store, crl) != 1)
            break;

        X509_CRL_free(crl);
        crl = NULL;
    }

    va_end(vaptr);

    /*很重要，因为在循环中有未释放的可能。*/
    if(crl)
        X509_CRL_free(crl);

    return count;
}

X509_STORE_CTX *abcdk_openssl_verify_crt_prepare(X509_STORE *store, X509 *crt)
{
    X509_STORE_CTX *ctx = NULL;
    int chk;

    assert(store != NULL && crt != NULL);

    ctx = X509_STORE_CTX_new();
    if (!ctx)
        return NULL;

    chk = X509_STORE_CTX_init(ctx, store, crt, NULL);
    if (chk == 1)
        return ctx;

final:

    if(ctx)
    {
        X509_STORE_CTX_cleanup(ctx);
        X509_STORE_CTX_free(ctx);
    }

    return NULL;
}

void abcdk_openssl_ssl_ctx_free(SSL_CTX **ctx)
{
    if (!ctx || !*ctx)
        return;

    SSL_CTX_free(*ctx);

    /*Set to NULL(0).*/
    *ctx = NULL;
}

SSL_CTX *abcdk_openssl_ssl_ctx_alloc(int server,const char *cafile,const char *capath,int crl_check)
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

    // SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
    // SSL_CTX_set_max_proto_version(ctx, TLS1_3_VERSION);

    if (cafile || capath)
    {
        chk = SSL_CTX_load_verify_locations(ctx, cafile, capath);
        if (chk != 1)
            goto final_error;
    }

#if ABCDK_VERSION_AT_LEAST((OPENSSL_VERSION_NUMBER >> 20), ((OPENSSL_VERSION_NUMBER >> 12) & 0xFF), 0x100, 0x02)

    X509_VERIFY_PARAM *param = SSL_CTX_get0_param(ctx);
    if(!param && crl_check)
        goto final_error;

    chk = X509_VERIFY_PARAM_set_purpose(param, X509_PURPOSE_ANY);
    if(chk != 1)
        goto final_error;

    if(crl_check == 2)
        chk = X509_VERIFY_PARAM_set_flags(param, X509_V_FLAG_CRL_CHECK | X509_V_FLAG_CRL_CHECK_ALL);
    else if(crl_check == 1)
        chk = X509_VERIFY_PARAM_set_flags(param, X509_V_FLAG_CRL_CHECK);
    else 
        chk = 1;

    if(chk != 1)
        goto final_error;

#endif //ABCDK_VERSION_AT_LEAST((OPENSSL_VERSION_NUMBER >> 20), ((OPENSSL_VERSION_NUMBER >> 12) & 0xFF), 0x100, 0x02)


    /*禁止会话复用。*/
    SSL_CTX_set_session_cache_mode(ctx, SSL_SESS_CACHE_OFF);
    
#ifdef SSL_OP_NO_TICKET
    /*禁用会话票据*/
    SSL_CTX_set_options(ctx, SSL_OP_NO_TICKET);
#endif //SSL_OP_NO_TICKET
    

    return ctx;

final_error:

    SSL_CTX_free(ctx);

    return NULL;
}

int abcdk_openssl_ssl_ctx_load_crt(SSL_CTX *ctx, const char *crt, const char *key, const char *pwd)
{
    int chk;

    assert(ctx != NULL);

    if (crt)
    {
        chk = SSL_CTX_use_certificate_file(ctx, crt, SSL_FILETYPE_PEM);
        if (chk != 1)
            ABCDK_ERRNO_AND_GOTO1(EINVAL, final_error);
    }

    if (key && pwd)
        SSL_CTX_set_default_passwd_cb_userdata(ctx, (void *)pwd);

    if (key)
    {
        chk = SSL_CTX_use_PrivateKey_file(ctx, key, SSL_FILETYPE_PEM);
        if (chk != 1)
            ABCDK_ERRNO_AND_GOTO1(EINVAL, final_error);
    }

    if (crt && key)
    {
        chk = SSL_CTX_check_private_key(ctx);
        if (chk != 1)
            ABCDK_ERRNO_AND_GOTO1(EINVAL, final_error);
    }

    return 0;

final_error:

    return -1;
}

SSL_CTX *abcdk_openssl_ssl_ctx_alloc_load(int server, const char *cafile, const char *capath, const char *crt, const char *key, const char *pwd)
{
    SSL_CTX *ctx = NULL;
    int chk;

    ctx = abcdk_openssl_ssl_ctx_alloc(server, cafile, capath, (capath ? 2 : 0));
    if (!ctx)
    {
        if(cafile)
            abcdk_trace_output(LOG_WARNING, "加载'%s'错误。\n",cafile);
        if(capath)
            abcdk_trace_output(LOG_WARNING, "加载'%s'错误。\n",capath);
            
        goto ERR;
    }

    if(server && !crt)
    {
        abcdk_trace_output(LOG_WARNING, "服务端的证书不能省略。\n");
        goto ERR;
    }

    if(!server && (cafile || capath) && !crt)
    {
        abcdk_trace_output(LOG_WARNING, "客户端的证书不能省略，因为CA证书或路径已经加载。\n");
        goto ERR;
    }

    chk = abcdk_openssl_ssl_ctx_load_crt(ctx, crt, key, pwd);
    if (chk != 0)
    {
        if(crt)
            abcdk_trace_output(LOG_WARNING, "加载'%s'错误。\n",crt);
        if(key)
            abcdk_trace_output(LOG_WARNING, "加载'%s'错误。\n",key);

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
            ABCDK_ERRNO_AND_GOTO1(EINVAL, final_error);

        if (server)
            SSL_set_accept_state(ssl);
        else
            SSL_set_connect_state(ssl);
    }

try_again:

    chk = SSL_do_handshake(ssl);
    if (chk == 1)
        goto final;

    err = SSL_get_error(ssl, chk);
    if (err == SSL_ERROR_WANT_WRITE)
    {
        chk = abcdk_poll(fd, 0x02, timeout);
        if (chk <= 0)
            ABCDK_ERRNO_AND_GOTO1(ETIMEDOUT, final_error);

        ABCDK_ERRNO_AND_GOTO1(0, try_again);
    }
    else if (err == SSL_ERROR_WANT_READ)
    {
        chk = abcdk_poll(fd, 0x01, timeout);
        if (chk <= 0)
            ABCDK_ERRNO_AND_GOTO1(ETIMEDOUT, final_error);

        ABCDK_ERRNO_AND_GOTO1(0, try_again);
    }
    else
    {
        ABCDK_ERRNO_AND_GOTO1(ENOSYS, final_error);
    }

final:

    return 0;

final_error:

    return -1;
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