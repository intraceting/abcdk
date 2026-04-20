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

EVP_PKEY *abcdk_openssl_pki_generate_pkey(int bits)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    EVP_PKEY *pkey = NULL;
    RSA *rsa = NULL;

    assert(bits > 0 && (bits % 2) == 0);

    rsa = abcdk_openssl_rsa_create(bits, RSA_F4);
    if (!rsa)
        return NULL;

    pkey = EVP_PKEY_new();
    if (!pkey)
    {
        abcdk_openssl_rsa_free(&rsa);
        return NULL;
    }

    EVP_PKEY_assign_RSA(pkey, rsa); // rsa被pkey托管.

    return pkey;
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

int abcdk_openssl_pki_check_cert_and_pkey(X509 *cert, EVP_PKEY *pri_pkey)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else  // #ifndef HAVE_OPENSSL
    int chk;

    assert(cert != NULL && pri_pkey != NULL);

    EVP_PKEY *pub_pkey = X509_get_pubkey(cert);

    // 返回值: 1 匹配, 0 不匹配, -1 错误, -2 不支持比较
    chk = EVP_PKEY_cmp(pub_pkey, pri_pkey);
    EVP_PKEY_free(pub_pkey); // free.

    if (chk == 1)
        return 0;
    else if (chk == 0)
        return -1;
    else if (chk == -2)
        return -2;
    else
        return -3;
#endif // #ifndef HAVE_OPENSSL
}

X509 *abcdk_openssl_pki_issue_cert(EVP_PKEY *pkey, ASN1_INTEGER *serial, const char *cn, const char *org, int ca_or_not, abcdk_option_t *opt,
                                   X509 *issuer_cert, EVP_PKEY *issuer_pkey)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else // #ifndef HAVE_OPENSSL
    X509 *cert = NULL;
    int chk;

    assert(serial != NULL && cn != NULL && org != NULL && opt != NULL);
    assert((issuer_cert != NULL && issuer_pkey != NULL) || (issuer_cert == NULL && issuer_pkey == NULL));
    assert(*cn != '\0' && *org != '\0');

    cert = X509_new();
    if (!cert)
        return NULL;

    if (issuer_cert != NULL && issuer_pkey != NULL)
    {
        chk = abcdk_openssl_pki_check_cert_and_pkey(issuer_cert, issuer_pkey);
        assert(chk != 0);
    }

    long version = abcdk_option_get_long(opt, "--version", 0, 2);
    long not_before_days = abcdk_option_get_long(opt, "--not-before-days", 0, 0);
    long not_after_days = abcdk_option_get_long(opt, "--not-before-days", 0, 30);
    long pathlen = abcdk_option_get_long(opt, "--pathlen", 0, 0);
    const char *san = abcdk_option_get(opt, "--san", 0, "DNS:localhost,DNS:localhost4,DNS:localhost6,IP:127.0.0.1,IP:::1");           // subject-alt-name
    const char *sigalg = abcdk_option_get(opt, "--sigalg", 0, "sha384"); // signature-algorithm

    X509_set_version(cert, version);
    X509_set_serialNumber(cert, serial);

    X509_gmtime_adj(X509_get_notBefore(cert), not_before_days * 24 * 3600);
    X509_gmtime_adj(X509_get_notAfter(cert), not_after_days * 24 * 3600);

    X509_set_pubkey(cert, pkey);

    if (issuer_cert)
    {
        X509_set_issuer_name(cert, X509_get_subject_name(issuer_cert));
    }
    else
    {
        X509_NAME *name = X509_get_subject_name(cert);
        X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC, (unsigned char *)cn, -1, -1, 0);
        X509_NAME_add_entry_by_txt(name, "O", MBSTRING_ASC, (unsigned char *)org, -1, -1, 0);

        X509_set_issuer_name(cert, name);
    }

    if (ca_or_not)
    {
        if(pathlen>0)
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

    // SKI
    _abcdk_openssl_pki_add_ext(cert, NID_subject_key_identifier, "hash");

    // AKI
    if (ca_or_not && issuer_pkey)
        _abcdk_openssl_pki_add_ext(cert, NID_authority_key_identifier, "keyid:always,issuer:always");
    else
        _abcdk_openssl_pki_add_ext(cert, NID_authority_key_identifier, "keyid:always");

    // 签名.
    if (abcdk_strcmp(sigalg, "sha384", 0) == 0)
        chk = X509_sign(cert, (issuer_pkey ? issuer_pkey : pkey), EVP_sha384());
    else if (abcdk_strcmp(sigalg, "sha512", 0) == 0)
        chk = X509_sign(cert, (issuer_pkey ? issuer_pkey : pkey), EVP_sha512());
#ifndef OPENSSL_NO_SM3
    else if (abcdk_strcmp(sigalg, "sm3", 0) == 0)
        chk = X509_sign(cert, (issuer_pkey ? issuer_pkey : pkey), EVP_sm3());
#endif // # ifndef OPENSSL_NO_SM3
    else
        chk = X509_sign(cert, (issuer_pkey ? issuer_pkey : pkey), EVP_sha256());

    if (chk <= 0) // 签名长度(字节).
    {
        X509_free(cert);
        return NULL;
    }

    return cert;

#endif // #ifndef HAVE_OPENSSL
}
