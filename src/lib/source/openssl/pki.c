/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2026 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/io.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/object.h"
#include "abcdk/util/dirent.h"
#include "abcdk/util/getpass.h"
#include "abcdk/util/string.h"
#include "abcdk/openssl/openssl.h"
#include "abcdk/openssl/pki.h"
#include "abcdk/openssl/util.h"

void abcdk_openssl_pki_destroy_key(EVP_PKEY **key)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return;
#else  // #ifndef HAVE_OPENSSL
    EVP_PKEY *key_p;

    if (!key || !*key)
        return;

    key_p = *key;
    *key = NULL;

    EVP_PKEY_free(key_p);
#endif // #ifndef HAVE_OPENSSL
}

EVP_PKEY *abcdk_openssl_pki_generate_key_from_rsa(int bits)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    EVP_PKEY_CTX *ctx = NULL;
    EVP_PKEY *key = NULL;

    ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, NULL);
    if (!ctx)
        return NULL;

    if (EVP_PKEY_keygen_init(ctx) <= 0)
        goto ERR;

    if (EVP_PKEY_CTX_set_rsa_keygen_bits(ctx, bits) <= 0)
        goto ERR;

    if (EVP_PKEY_keygen(ctx, &key) <= 0)
        goto ERR;

    EVP_PKEY_CTX_free(ctx);
    return key;

ERR:
    if (ctx)
        EVP_PKEY_CTX_free(ctx);
    if (key)
        EVP_PKEY_free(key);
    return NULL;
#endif // #ifndef HAVE_OPENSSL
}

EVP_PKEY *abcdk_openssl_pki_generate_key_from_ec(int nid)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    EVP_PKEY_CTX *ctx = NULL;
    EVP_PKEY *key = NULL;

    ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_EC, NULL);
    if (!ctx)
        return NULL;

    if (EVP_PKEY_keygen_init(ctx) <= 0)
        goto ERR;

    if (EVP_PKEY_CTX_set_ec_paramgen_curve_nid(ctx, nid) <= 0)
        goto ERR;

    if (EVP_PKEY_keygen(ctx, &key) <= 0)
        goto ERR;

    EVP_PKEY_CTX_free(ctx);
    return key;

ERR:
    if (ctx)
        EVP_PKEY_CTX_free(ctx);
    if (key)
        EVP_PKEY_free(key);
    return NULL;
#endif // #ifndef HAVE_OPENSSL
}

int abcdk_openssl_pki_private_key_was(EVP_PKEY *key)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    uint8_t *prikey_buf = NULL;
    int prikey_len = 0;

    assert(key != NULL);

    /*检查是否存在私钥数据.*/
    prikey_len = i2d_PrivateKey(key, &prikey_buf);
    abcdk_openssl_free((void **)&prikey_buf); // free.

    return (prikey_len > 0 ? 1 : 0);
#endif // #ifndef HAVE_OPENSSL
}

EVP_PKEY *abcdk_openssl_pki_generate_key_to_public(EVP_PKEY *key)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    int prikey_chk = 0;
    uint8_t *pubder_buf = NULL;
    const uint8_t *pubder_buf_ptr = NULL;
    int pubder_len = 0;
    EVP_PKEY *pubkey = NULL;

    assert(key != NULL);

    /*检查是否存在私钥数据.*/
    prikey_chk = abcdk_openssl_pki_private_key_was(key);
    if (!prikey_chk)
        return NULL;

    pubder_len = i2d_PUBKEY(key, &pubder_buf);
    if (pubder_len <= 0)
        return NULL;

    pubder_buf_ptr = pubder_buf; // 复制指针, 因为下面的函数会修改传入的指针.
    pubkey = d2i_PUBKEY(NULL, &pubder_buf_ptr, pubder_len);
    abcdk_openssl_free((void **)&pubder_buf); // free.

    return pubkey;
#endif // #ifndef HAVE_OPENSSL
}

typedef struct _abcdk_openssl_pki_pem_password_ctx
{
    void *opaque;
    abcdk_get_password_cb get_password_cb;
} abcdk_openssl_pki_pem_password_ctx_t;

static int _abcdk_openssl_pki_pem_password_cb(char *buf, int size, int rwflag, void *userdata)
{
    abcdk_openssl_pki_pem_password_ctx_t *ctx = (abcdk_openssl_pki_pem_password_ctx_t *)userdata;
    int chk;

    return ctx->get_password_cb(buf, size, rwflag == 1, ctx->opaque);
}

abcdk_object_t *abcdk_openssl_pki_export_key(EVP_PKEY *key, abcdk_get_password_cb get_password_cb, void *opaque)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    BIO *bp;
    long data_l;
    void *data_p;
    abcdk_object_t *obj = NULL;
    int prikey_chk = 0;
    int chk;

    assert(key != NULL);

    /*检查是否存在私钥数据.*/
    prikey_chk = abcdk_openssl_pki_private_key_was(key);

    bp = BIO_new(BIO_s_mem());
    if (!bp)
        return NULL;

    if (prikey_chk)
    {
        if (get_password_cb)
        {
            abcdk_openssl_pki_pem_password_ctx_t ctx = {opaque, get_password_cb};
            chk = PEM_write_bio_PrivateKey(bp, key, EVP_aes_256_cbc(), NULL, 0, _abcdk_openssl_pki_pem_password_cb, &ctx);
        }
        else if(opaque)
        {
            chk = PEM_write_bio_PrivateKey(bp, key, EVP_aes_256_cbc(), NULL, 0, NULL, opaque);
        }
        else 
        {
            chk = PEM_write_bio_PrivateKey(bp, key, EVP_aes_256_cbc(), NULL, 0, NULL, NULL);
        }
    }
    else
    {
        chk = PEM_write_bio_PUBKEY(bp, key);
    }

    data_l = BIO_get_mem_data(bp, &data_p);
    obj = abcdk_object_copyfrom(data_p, data_l);
    BIO_free(bp);

    return (obj);
#endif // #ifndef HAVE_OPENSSL
}

EVP_PKEY *abcdk_openssl_pki_import_key(const char *data, size_t size, int pri_or_pub, abcdk_get_password_cb get_password_cb, void *opaque)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return;
#else  // #ifndef HAVE_OPENSSL
    EVP_PKEY *key = NULL;
    FILE *fp;

    assert(data != NULL && size > 0);

    fp = fmemopen((void *)data, size, "r");
    if (!fp)
        return NULL;

    key = abcdk_openssl_pki_import_key_from_fp(fp, pri_or_pub, get_password_cb, opaque);
    fclose(fp);

    return key;
#endif // #ifndef HAVE_OPENSSL
}

EVP_PKEY *abcdk_openssl_pki_import_key_from_file(const char *file, int pri_or_pub, abcdk_get_password_cb get_password_cb, void *opaque)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return;
#else  // #ifndef HAVE_OPENSSL
    EVP_PKEY *key = NULL;
    FILE *fp;

    assert(file != NULL);

    fp = fopen(file, "r");
    if (!fp)
        return NULL;

    key = abcdk_openssl_pki_import_key_from_fp(fp, pri_or_pub, get_password_cb, opaque);
    fclose(fp);

    return key;
#endif // #ifndef HAVE_OPENSSL
}

EVP_PKEY *abcdk_openssl_pki_import_key_from_fp(FILE *fp, int pri_or_pub, abcdk_get_password_cb get_password_cb, void *opaque)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return;
#else  // #ifndef HAVE_OPENSSL
    EVP_PKEY *key = NULL;

    assert(fp != NULL);

    if (get_password_cb)
    {
        abcdk_openssl_pki_pem_password_ctx_t ctx = {opaque, get_password_cb};
        if (pri_or_pub)
            key = PEM_read_PrivateKey(fp, NULL, _abcdk_openssl_pki_pem_password_cb, &ctx);
        else
            key = PEM_read_PUBKEY(fp, NULL, _abcdk_openssl_pki_pem_password_cb, &ctx);
    }
    else
    {
        if (pri_or_pub)
            key = PEM_read_PrivateKey(fp, NULL, NULL, opaque);
        else
            key = PEM_read_PUBKEY(fp, NULL, NULL, opaque);
    }

    return key;
#endif // #ifndef HAVE_OPENSSL
}

void abcdk_openssl_pki_destroy_serial(ASN1_INTEGER **serial)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return;
#else  // #ifndef HAVE_OPENSSL
    ASN1_INTEGER *serial_p;

    if (!serial || !*serial)
        return;

    serial_p = *serial;
    *serial = NULL;

    ASN1_INTEGER_free(serial_p);
#endif // #ifndef HAVE_OPENSSL
}

ASN1_INTEGER *abcdk_openssl_pki_generate_serial(int bits)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    BIGNUM *bn = NULL;
    ASN1_INTEGER *serial = NULL;

    assert(bits > 0 && (bits % 2) == 0);

    bn = BN_new();
    if (!bn)
        return NULL;

    /*
     * 随机生成.
     * RFC要求: 最高位不能全是1, 避免负数.
     */
    BN_rand(bn, bits, BN_RAND_TOP_ANY, BN_RAND_BOTTOM_ANY);

    serial = BN_to_ASN1_INTEGER(bn, NULL);
    abcdk_openssl_bn_free(&bn);

    return serial;
#endif // #ifndef HAVE_OPENSSL
}

abcdk_object_t *abcdk_openssl_pki_string_serial(ASN1_INTEGER *ai, int hex_or_dec)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    BIGNUM *bn = NULL;
    char *str = NULL;
    abcdk_object_t *obj_str = NULL;

    assert(ai != NULL);

    bn = ASN1_INTEGER_to_BN(ai, NULL);
    if (!bn)
        return NULL;

    str = (hex_or_dec ? BN_bn2hex(bn) : BN_bn2dec(bn));
    BN_free(bn); // free.

    if (!str)
        return NULL;

    obj_str = abcdk_object_copyfrom(str, strlen(str));
    OPENSSL_free(str);

    return obj_str;
#endif // #ifndef HAVE_OPENSSL
}

void abcdk_openssl_pki_destroy_cert(X509 **cert)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return;
#else  // #ifndef HAVE_OPENSSL
    X509 *cert_p;

    if (!cert || !*cert)
        return;

    cert_p = *cert;
    *cert = NULL;

    X509_free(cert_p);
#endif // #ifndef HAVE_OPENSSL
}

#ifdef HAVE_OPENSSL
/**
 * @return 1 成功, 0 失败.
 */
static int _abcdk_openssl_pki_add_ext(X509 *cert, int nid, const char *value, ...)
{
    char buf[8000] = {0};
    int chk;

    va_list ap;
    va_start(ap, value);
    vsnprintf(buf, 8000, value, ap);
    va_end(ap);

    X509_EXTENSION *ex_info = X509V3_EXT_conf_nid(NULL, NULL, nid, buf);
    if (!ex_info)
        return 0;

    chk = X509_add_ext(cert, ex_info, -1);
    X509_EXTENSION_free(ex_info);

    return chk;
}
#endif // #ifdef HAVE_OPENSSL

X509 *abcdk_openssl_pki_generate_cert(EVP_PKEY *pkey, ASN1_INTEGER *serial, const char *name_cn, const char *name_o, abcdk_option_t *opt,
                                      X509 *issuer_cert, EVP_PKEY *issuer_key)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else // #ifndef HAVE_OPENSSL
    X509 *cert = NULL;
    int chk;

    assert(serial != NULL && name_cn != NULL && name_o != NULL && opt != NULL);
    assert(*name_cn != '\0' && *name_o != '\0');
    assert((issuer_cert != NULL && issuer_key != NULL) || (issuer_cert == NULL && issuer_key == NULL));

    cert = X509_new();
    if (!cert)
        return NULL;

    if (issuer_cert != NULL && issuer_key != NULL)
    {
        chk = X509_check_private_key(issuer_cert, issuer_key);
        ABCDK_TRACE_ASSERT(chk == 1, ABCDK_GETTEXT("签发者的证书与私钥不匹配."));
    }

    int ca_or_not = abcdk_option_get_long(opt, "--ca", 0, 0); // Certificate Authority
    long version = abcdk_option_get_long(opt, "--version", 0, 2);
    long not_before_days = abcdk_option_get_long(opt, "--not-before-days", 0, 0);
    long not_after_days = abcdk_option_get_long(opt, "--not-after-days", 0, 30);
    long pathlen = abcdk_option_get_long(opt, "--pathlen", 0, 0);
    const char *name_c = abcdk_option_get(opt, "--name-c", 0, NULL);
    const char *name_co = abcdk_option_get(opt, "--name-co", 0, NULL);
    const char *san = abcdk_option_get(opt, "--san", 0, "DNS:localhost,DNS:localhost4,DNS:localhost6,IP:127.0.0.1,IP:::1"); // subject-alt-name
    const char *sigalg = abcdk_option_get(opt, "--sigalg", 0, "sha384");                                                    // signature-algorithm
    const char *crl = abcdk_option_get(opt, "--crl", 0, NULL);

    X509_set_version(cert, version);
    X509_set_serialNumber(cert, serial);

    X509_gmtime_adj(X509_get_notBefore(cert), not_before_days * 24 * 3600);
    X509_gmtime_adj(X509_get_notAfter(cert), not_after_days * 24 * 3600);

    X509_set_pubkey(cert, pkey);

    X509_NAME *name = X509_get_subject_name(cert);

    X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC, (unsigned char *)name_cn, -1, -1, 0);
    X509_NAME_add_entry_by_txt(name, "O", MBSTRING_ASC, (unsigned char *)name_o, -1, -1, 0);

    if (name_c)
        X509_NAME_add_entry_by_txt(name, "C", MBSTRING_ASC, (unsigned char *)name_c, -1, -1, 0);
    if (name_co)
        X509_NAME_add_entry_by_txt(name, "CO", MBSTRING_ASC, (unsigned char *)name_co, -1, -1, 0);

    if (issuer_cert)
        X509_set_issuer_name(cert, X509_get_subject_name(issuer_cert));
    else
        X509_set_issuer_name(cert, name);

    if (ca_or_not)
    {
        if (pathlen > 0)
            _abcdk_openssl_pki_add_ext(cert, NID_basic_constraints, "critical,CA:TRUE,pathlen:%d", pathlen);
        else
            _abcdk_openssl_pki_add_ext(cert, NID_basic_constraints, "critical,CA:TRUE");

        _abcdk_openssl_pki_add_ext(cert, NID_key_usage, "critical,keyCertSign,cRLSign");
    }
    else
    {
        _abcdk_openssl_pki_add_ext(cert, NID_basic_constraints, "CA:FALSE");
        _abcdk_openssl_pki_add_ext(cert, NID_key_usage, "digitalSignature,keyEncipherment");
        _abcdk_openssl_pki_add_ext(cert, NID_ext_key_usage, "serverAuth,clientAuth");

        if (san)
        {
            _abcdk_openssl_pki_add_ext(cert, NID_subject_alt_name, san);
        }
    }

    if (crl != NULL && issuer_cert != NULL && issuer_key != NULL)
    {
        _abcdk_openssl_pki_add_ext(cert, NID_crl_distribution_points, crl);
    }

    // SKI
    _abcdk_openssl_pki_add_ext(cert, NID_subject_key_identifier, "hash");

    // AKI
    if (issuer_key)
        _abcdk_openssl_pki_add_ext(cert, NID_authority_key_identifier, "keyid:always,issuer:always");
    else
        _abcdk_openssl_pki_add_ext(cert, NID_authority_key_identifier, "keyid:always");

    // 签名.
    if (abcdk_strcmp(sigalg, "sha384", 0) == 0)
        chk = X509_sign(cert, (issuer_key ? issuer_key : pkey), EVP_sha384());
    else if (abcdk_strcmp(sigalg, "sha512", 0) == 0)
        chk = X509_sign(cert, (issuer_key ? issuer_key : pkey), EVP_sha512());
#ifndef OPENSSL_NO_SM3
    else if (abcdk_strcmp(sigalg, "sm3", 0) == 0)
        chk = X509_sign(cert, (issuer_key ? issuer_key : pkey), EVP_sm3());
#endif // # ifndef OPENSSL_NO_SM3
    else
        chk = X509_sign(cert, (issuer_key ? issuer_key : pkey), EVP_sha256());

    if (chk <= 0) // 签名长度(字节).
    {
        X509_free(cert);
        return NULL;
    }

    return cert;

#endif // #ifndef HAVE_OPENSSL
}

abcdk_object_t *abcdk_openssl_pki_export_cert(X509 *cert)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    BIO *bp;
    long data_l;
    void *data_p;
    abcdk_object_t *obj = NULL;
    int chk;

    assert(cert != NULL);

    bp = BIO_new(BIO_s_mem());
    if (!bp)
        return NULL;

    chk = PEM_write_bio_X509(bp, cert);

    data_l = BIO_get_mem_data(bp, &data_p);
    obj = abcdk_object_copyfrom(data_p, data_l);
    BIO_free(bp);

    return (obj);
#endif // #ifndef HAVE_OPENSSL
}

X509 *abcdk_openssl_pki_import_cert(const char *data, size_t size)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL

    X509 *cert = NULL;
    FILE *fp = NULL;

    assert(data != NULL && size > 0);

    fp = fmemopen((void *)data, size, "r");
    if (!fp)
        return NULL;

    cert = abcdk_openssl_pki_import_cert_from_fp(fp);
    fclose(fp);

    return cert;
#endif // #ifndef HAVE_OPENSSL
}

X509 *abcdk_openssl_pki_import_cert_from_file(const char *file)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    X509 *cert = NULL;
    FILE *fp = NULL;

    assert(file != NULL);

    fp = fopen(file, "r");
    if (!fp)
        return NULL;

    cert = abcdk_openssl_pki_import_cert_from_fp(fp);
    fclose(fp);

    return cert;
#endif // #ifndef HAVE_OPENSSL
}

X509 *abcdk_openssl_pki_import_cert_from_fp(FILE *fp)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    X509 *cert = NULL;

    assert(fp != NULL);

    cert = PEM_read_X509(fp, NULL, NULL, NULL);

    return cert;
#endif // #ifndef HAVE_OPENSSL
}

void abcdk_openssl_pki_destroy_crl(X509_CRL **crl)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return;
#else  // #ifndef HAVE_OPENSSL
    X509_CRL *crl_p;

    if (!crl || !*crl)
        return;

    crl_p = *crl;
    *crl = NULL;

    X509_CRL_free(crl_p);
#endif // #ifndef HAVE_OPENSSL
}

int abcdk_openssl_pki_cert_is_revoked(X509_CRL *crl, X509 *cert)
{
    X509_NAME *cert_issuer = X509_get_issuer_name(cert);
    X509_NAME *crl_issuer = X509_CRL_get_issuer(crl);

    int chk = X509_NAME_cmp(cert_issuer, crl_issuer);
    ABCDK_TRACE_ASSERT(chk == 0, ABCDK_GETTEXT("待吊销证书与吊销列表应当来自同一个签发者."));

    const ASN1_INTEGER *serial = X509_get_serialNumber(cert);

    STACK_OF(X509_REVOKED) *rev_list = X509_CRL_get_REVOKED(crl);
    if (!rev_list)
        return 0;

    for (int i = 0; i < sk_X509_REVOKED_num(rev_list); i++)
    {
        X509_REVOKED *rev = sk_X509_REVOKED_value(rev_list, i);

        const ASN1_INTEGER *rev_serial = X509_REVOKED_get0_serialNumber(rev);

        if (ASN1_INTEGER_cmp(serial, rev_serial) == 0)
            return 1; // 已吊销
    }

    return 0;
}

int abcdk_openssl_pki_revoke_cert(X509_CRL **crl, X509 *cert, int reason_code, X509 *issuer_cert, EVP_PKEY *issuer_key)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return -1;
#else // #ifndef HAVE_OPENSSL
    X509_CRL *crl_p;
    int chk;

    assert(crl != NULL && issuer_cert != NULL && issuer_key != NULL);

    crl_p = *crl;

    chk = X509_check_private_key(issuer_cert, issuer_key);
    ABCDK_TRACE_ASSERT(chk == 1, ABCDK_GETTEXT("签发者的证书与私钥不匹配."));

    if (!crl_p)
    {
        crl_p = X509_CRL_new();
        if (!crl_p)
            return -1;

        X509_CRL_set_version(crl_p, 1);
        X509_CRL_set_issuer_name(crl_p, X509_get_subject_name(issuer_cert)); // 绑定签发者.

        *crl = crl_p; // copy.
    }
    else
    {
        X509_NAME *issuer_subject = X509_get_subject_name(issuer_cert);
        X509_NAME *crl_issuer = X509_CRL_get_issuer(crl_p);

        chk = X509_NAME_cmp(issuer_subject, crl_issuer);
        ABCDK_TRACE_ASSERT(chk == 0, ABCDK_GETTEXT("吊销列表必须由签发者创建才允许吊销证书."));
    }

    if (!cert)
        return 0;

    ASN1_INTEGER *issuer_serial = X509_get_serialNumber(issuer_cert);
    ASN1_INTEGER *cert_serial = X509_get_serialNumber(cert);

    chk = ASN1_INTEGER_cmp(issuer_serial, cert_serial);
    ABCDK_TRACE_ASSERT(chk != 0, ABCDK_GETTEXT("签发者不允许吊销自己."));

    chk = abcdk_openssl_pki_cert_is_revoked(crl_p, cert);
    if (chk != 0)
        return -2;

    X509_REVOKED *rev = X509_REVOKED_new();

    X509_REVOKED_set_serialNumber(rev, X509_get_serialNumber(cert)); // add sn

    ASN1_TIME *rev_tm = ASN1_TIME_new();
    ASN1_TIME_set(rev_tm, time(NULL));
    ASN1_TIME_free(rev_tm); // free.

    X509_REVOKED_set_revocationDate(rev, rev_tm); // add time.

    ASN1_ENUMERATED *rev_reason = ASN1_ENUMERATED_new();
    ASN1_ENUMERATED_set(rev_reason, reason_code);

    X509_EXTENSION *rev_ext = X509_EXTENSION_create_by_NID(NULL, NID_crl_reason, 0, (ASN1_OCTET_STRING *)rev_reason);
    ASN1_ENUMERATED_free(rev_reason); // free.

    X509_REVOKED_add_ext(rev, rev_ext, -1); // add reason
    X509_EXTENSION_free(rev_ext);           // free.

    X509_CRL_add0_revoked(crl_p, rev);

    return 0;

#endif // #ifndef HAVE_OPENSSL
}

int abcdk_openssl_pki_update_crl(X509_CRL *crl, abcdk_option_t *opt, X509 *issuer_cert, EVP_PKEY *issuer_key)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return -1;
#else // #ifndef HAVE_OPENSSL
    int chk;

    assert(crl != NULL && opt != NULL && issuer_cert != NULL && issuer_key != NULL);

    chk = X509_check_private_key(issuer_cert, issuer_key);
    ABCDK_TRACE_ASSERT(chk == 1, ABCDK_GETTEXT("签发者的证书与私钥不匹配."));

    X509_NAME *issuer_subject = X509_get_subject_name(issuer_cert);
    X509_NAME *crl_issuer = X509_CRL_get_issuer(crl);

    chk = X509_NAME_cmp(issuer_subject, crl_issuer);
    ABCDK_TRACE_ASSERT(chk == 0, ABCDK_GETTEXT("吊销列表必须由签发者创建才允许更新."));

    long last_update_days = abcdk_option_get_long(opt, "--last-update-days", 0, 0);
    long next_update_days = abcdk_option_get_long(opt, "--next-update-days", 0, 30);
    const char *sigalg = abcdk_option_get(opt, "--sigalg", 0, "sha384"); // signature-algorithm

#if OPENSSL_VERSION_NUMBER >= 0x10100000L
    ASN1_TIME *last = ASN1_TIME_new();
    ASN1_TIME *next = ASN1_TIME_new();

    ASN1_TIME_adj(last, time(NULL), last_update_days, 0);
    ASN1_TIME_adj(next, time(NULL), next_update_days, 0);

    X509_CRL_set1_lastUpdate(crl, last);
    X509_CRL_set1_nextUpdate(crl, next);

    ASN1_TIME_free(last);
    ASN1_TIME_free(next);
#else  // #if OPENSSL_VERSION_NUMBER >= 0x10100000L
    X509_gmtime_adj(X509_CRL_get_lastUpdate(crl), last_update_days * 24 * 3600);
    X509_gmtime_adj(X509_CRL_get_nextUpdate(crl), next_update_days * 24 * 3600);
#endif // #if OPENSSL_VERSION_NUMBER >= 0x10100000L

    // 排序.
    X509_CRL_sort(crl);

    // 签名.
    if (abcdk_strcmp(sigalg, "sha384", 0) == 0)
        chk = X509_CRL_sign(crl, issuer_key, EVP_sha384());
    else if (abcdk_strcmp(sigalg, "sha512", 0) == 0)
        chk = X509_CRL_sign(crl, issuer_key, EVP_sha512());
#ifndef OPENSSL_NO_SM3
    else if (abcdk_strcmp(sigalg, "sm3", 0) == 0)
        chk = X509_CRL_sign(crl, issuer_key, EVP_sm3());
#endif // # ifndef OPENSSL_NO_SM3
    else
        chk = X509_CRL_sign(crl, issuer_key, EVP_sha256());

    if (chk <= 0) // 签名长度(字节).
        return -1;

    return 0;
#endif // #ifndef HAVE_OPENSSL
}

abcdk_object_t *abcdk_openssl_pki_export_crl(X509_CRL *crl)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    BIO *bp;
    long data_l;
    void *data_p;
    abcdk_object_t *obj = NULL;
    int chk;

    assert(crl != NULL);

    bp = BIO_new(BIO_s_mem());
    if (!bp)
        return NULL;

    chk = PEM_write_bio_X509_CRL(bp, crl);

    data_l = BIO_get_mem_data(bp, &data_p);
    obj = abcdk_object_copyfrom(data_p, data_l);
    BIO_free(bp);

    return (obj);
#endif // #ifndef HAVE_OPENSSL
}

X509_CRL *abcdk_openssl_pki_import_crl(const char *data, size_t size)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    X509_CRL *ctx = NULL;
    FILE *fp = NULL;

    assert(data != NULL && size > 0);

    fp = fmemopen((void *)data, size, "r");
    if (!fp)
        return NULL;

    ctx = abcdk_openssl_pki_import_crl_from_fp(fp);
    fclose(fp);

    return ctx;
#endif // #ifndef HAVE_OPENSSL
}

X509_CRL *abcdk_openssl_pki_import_crl_from_file(const char *file)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    X509_CRL *ctx = NULL;
    FILE *fp = NULL;

    assert(file != NULL);

    fp = fopen(file, "r");
    if (!fp)
        return NULL;

    ctx = abcdk_openssl_pki_import_crl_from_fp(fp);
    fclose(fp);

    return ctx;
#endif // #ifndef HAVE_OPENSSL
}

X509_CRL *abcdk_openssl_pki_import_crl_from_fp(FILE *fp)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    X509_CRL *ctx = NULL;

    assert(fp != NULL);

    ctx = PEM_read_X509_CRL(fp, NULL, NULL, NULL);

    return ctx;
#endif // #ifndef HAVE_OPENSSL
}

STACK_OF(X509) * abcdk_openssl_pki_import_cert_chain(const char *data, size_t size)
{
}

STACK_OF(X509) * abcdk_openssl_pki_import_cert_chain_from_file(const char *file)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    STACK_OF(X509) * cert_chain;
    FILE *fp;

    assert(file != NULL);

    fp = fopen(file, "r");
    if (!fp)
        return NULL;

    cert_chain = abcdk_openssl_pki_import_cert_chain_from_fp(fp);
    fclose(fp);

    return cert_chain;
#endif // #ifndef HAVE_OPENSSL
}

STACK_OF(X509) * abcdk_openssl_pki_import_cert_chain_from_fp(FILE *fp)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    STACK_OF(X509) * cert_chain;
    X509 *cert = NULL;

    assert(fp != NULL);

    cert_chain = sk_X509_new_null();
    if (!cert_chain)
        return NULL;

    /*循环读取所以证书.*/
    while (cert = PEM_read_X509(fp, NULL, NULL, NULL))
        sk_X509_push(cert_chain, cert);

    return cert_chain;
#endif // #ifndef HAVE_OPENSSL
}
