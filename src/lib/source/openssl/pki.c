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

abcdk_object_t *abcdk_openssl_pki_export_pkey(EVP_PKEY *pkey, int pubkey, uint8_t *passwd, int passwd_len)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else // #ifndef HAVE_OPENSSL
    BIO *bp;
    long data_l;
    void *data_p;
    abcdk_object_t *obj = NULL;
    int chk;

    assert(pkey != NULL);

    bp = BIO_new(BIO_s_mem());
    if(!bp)
        return NULL;

    if(pubkey)
        chk = PEM_write_bio_PUBKEY(bp, pkey);
    else 
        chk = PEM_write_bio_PrivateKey(bp, pkey, EVP_aes_256_cbc(), passwd, passwd_len, NULL, NULL);

    data_l = BIO_get_mem_data(bp, &data_p);
    obj = abcdk_object_copyfrom(data_p,data_l);        
    BIO_free(bp);

    return (obj);
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
    if (!str)
    {
        BN_free(bn);
        return NULL;
    }

    obj_str = abcdk_object_copyfrom(str, strlen(str));
    OPENSSL_free(str);

    return obj_str;
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
                                      X509 *issuer_cert, EVP_PKEY *issuer_pkey)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else // #ifndef HAVE_OPENSSL
    X509 *cert = NULL;
    int chk;

    assert(serial != NULL && name_cn != NULL && name_o != NULL && opt != NULL);
    assert(*name_cn != '\0' && *name_o != '\0');
    assert((issuer_cert != NULL && issuer_pkey != NULL) || (issuer_cert == NULL && issuer_pkey == NULL));
    

    cert = X509_new();
    if (!cert)
        return NULL;

    if (issuer_cert != NULL && issuer_pkey != NULL)
    {
        chk = X509_check_private_key(issuer_cert, issuer_pkey);
        ABCDK_TRACE_ASSERT(chk == 1,ABCDK_GETTEXT("签发者的证书与私钥不匹配."));
    }

    int ca_or_not = abcdk_option_get_long(opt, "--ca", 0, 0);//Certificate Authority 
    long version = abcdk_option_get_long(opt, "--version", 0, 2);
    long not_before_days = abcdk_option_get_long(opt, "--not-before-days", 0, 0);
    long not_after_days = abcdk_option_get_long(opt, "--not-after-days", 0, 30);
    long pathlen = abcdk_option_get_long(opt, "--pathlen", 0, 0);
    const char *name_c = abcdk_option_get(opt, "--name-c", 0, NULL); 
    const char *name_co = abcdk_option_get(opt, "--name-co", 0, NULL); 
    const char *san = abcdk_option_get(opt, "--san", 0, "DNS:localhost,DNS:localhost4,DNS:localhost6,IP:127.0.0.1,IP:::1");// subject-alt-name
    const char *sigalg = abcdk_option_get(opt, "--sigalg", 0, "sha384"); // signature-algorithm
    const char *crl = abcdk_option_get(opt, "--crl",0, NULL);

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

    if(crl != NULL && issuer_cert != NULL && issuer_pkey != NULL)
    {
        _abcdk_openssl_pki_add_ext(cert, NID_crl_distribution_points, crl);
    }


    // SKI
    _abcdk_openssl_pki_add_ext(cert, NID_subject_key_identifier, "hash");

    // AKI
    if (issuer_pkey)
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

abcdk_object_t *abcdk_openssl_pki_export_cert(X509 *cert)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else // #ifndef HAVE_OPENSSL
    BIO *bp;
    long data_l;
    void *data_p;
    abcdk_object_t *obj = NULL;
    int chk;

    assert(cert != NULL);

    bp = BIO_new(BIO_s_mem());
    if(!bp)
        return NULL;

    chk = PEM_write_bio_X509(bp, cert);

    data_l = BIO_get_mem_data(bp, &data_p);
    obj = abcdk_object_copyfrom(data_p,data_l);        
    BIO_free(bp);

    return (obj);
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

int abcdk_openssl_pki_revoke_cert(X509_CRL **crl, X509 *cert, int reason_code, X509 *issuer_cert, EVP_PKEY *issuer_pkey)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return -1;
#else // #ifndef HAVE_OPENSSL
    X509_CRL *crl_p; 
    int chk;

    assert(crl != NULL && issuer_cert != NULL && issuer_pkey != NULL);

    crl_p = *crl;

    chk = X509_check_private_key(issuer_cert, issuer_pkey);
    ABCDK_TRACE_ASSERT(chk == 1,ABCDK_GETTEXT("签发者的证书与私钥不匹配."));

    if (!crl_p)
    {
        crl_p = X509_CRL_new();
        if (!crl_p)
            return -1;

        X509_CRL_set_version(crl_p, 1);
        X509_CRL_set_issuer_name(crl_p, X509_get_subject_name(issuer_cert));//绑定签发者.

        *crl = crl_p;// copy.
    }
    else
    {
        X509_NAME *issuer_subject = X509_get_subject_name(issuer_cert);
        X509_NAME *crl_issuer = X509_CRL_get_issuer(crl_p);

        chk = X509_NAME_cmp(issuer_subject, crl_issuer);
        ABCDK_TRACE_ASSERT(chk == 0,ABCDK_GETTEXT("吊销列表必须由签发者创建才允许吊销证书."));
    }

    if(!cert)
        return 0;

    ASN1_INTEGER *issuer_serial = X509_get_serialNumber(issuer_cert);
    ASN1_INTEGER *cert_serial = X509_get_serialNumber(cert);
    
    chk = ASN1_INTEGER_cmp(issuer_serial, cert_serial);
    ABCDK_TRACE_ASSERT(chk != 0,ABCDK_GETTEXT("签发者不允许吊销自己."));

    chk = abcdk_openssl_pki_cert_is_revoked(crl_p, cert);
    if (chk != 0)
        return -2;

    X509_REVOKED *rev = X509_REVOKED_new();

    X509_REVOKED_set_serialNumber(rev, X509_get_serialNumber(cert));//add sn

    ASN1_TIME *rev_tm = ASN1_TIME_new();
    ASN1_TIME_set(rev_tm, time(NULL));

    X509_REVOKED_set_revocationDate(rev, rev_tm);//add time.

    ASN1_ENUMERATED *rev_reason = ASN1_ENUMERATED_new();
    ASN1_ENUMERATED_set(rev_reason, reason_code);

    X509_EXTENSION *rev_ext = X509_EXTENSION_create_by_NID(NULL, NID_crl_reason, 0, (ASN1_OCTET_STRING *)rev_reason);

    X509_REVOKED_add_ext(rev, rev_ext, -1);// add reason

    X509_CRL_add0_revoked(crl_p, rev);

    return 0;

#endif // #ifndef HAVE_OPENSSL   
}

int abcdk_openssl_pki_update_crl(X509_CRL *crl, abcdk_option_t *opt, X509 *issuer_cert, EVP_PKEY *issuer_pkey)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return -1;
#else // #ifndef HAVE_OPENSSL
    int chk;

    assert(crl != NULL && opt != NULL && issuer_cert != NULL && issuer_pkey != NULL);

    chk = X509_check_private_key(issuer_cert, issuer_pkey);
    ABCDK_TRACE_ASSERT(chk == 1,ABCDK_GETTEXT("签发者的证书与私钥不匹配."));

    X509_NAME *issuer_subject = X509_get_subject_name(issuer_cert);
    X509_NAME *crl_issuer = X509_CRL_get_issuer(crl);

    chk = X509_NAME_cmp(issuer_subject, crl_issuer);
    ABCDK_TRACE_ASSERT(chk == 0, ABCDK_GETTEXT("吊销列表必须由签发者创建才允许更新."));

    long last_update_days = abcdk_option_get_long(opt, "--last-update-days", 0, 0);
    long next_update_days = abcdk_option_get_long(opt, "--next-update-days", 0, 30);
    const char *sigalg = abcdk_option_get(opt, "--sigalg", 0, "sha384"); // signature-algorithm

    ASN1_TIME *last = ASN1_TIME_new();
    ASN1_TIME *next = ASN1_TIME_new();

    ASN1_TIME_adj(last, time(NULL), last_update_days, 0);
    ASN1_TIME_adj(next, time(NULL), next_update_days, 0);

    X509_CRL_set1_lastUpdate(crl, last);
    X509_CRL_set1_nextUpdate(crl, next);

    ASN1_TIME_free(last);
    ASN1_TIME_free(next);

    // X509_gmtime_adj(X509_CRL_get_lastUpdate(crl), last_update_days * 24 * 3600);
    // X509_gmtime_adj(X509_CRL_get_nextUpdate(crl), next_update_days * 24 * 3600);

    // 排序.
    X509_CRL_sort(crl);
    
    // 签名.
    if (abcdk_strcmp(sigalg, "sha384", 0) == 0)
        chk = X509_CRL_sign(crl, issuer_pkey, EVP_sha384());
    else if (abcdk_strcmp(sigalg, "sha512", 0) == 0)
        chk = X509_CRL_sign(crl, issuer_pkey, EVP_sha512());
#ifndef OPENSSL_NO_SM3
    else if (abcdk_strcmp(sigalg, "sm3", 0) == 0)
        chk = X509_CRL_sign(crl, issuer_pkey, EVP_sm3());
#endif // # ifndef OPENSSL_NO_SM3
    else
        chk = X509_CRL_sign(crl, issuer_pkey, EVP_sha256());

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
#else // #ifndef HAVE_OPENSSL
    BIO *bp;
    long data_l;
    void *data_p;
    abcdk_object_t *obj = NULL;
    int chk;

    assert(crl != NULL);

    bp = BIO_new(BIO_s_mem());
    if(!bp)
        return NULL;

    chk = PEM_write_bio_X509_CRL(bp, crl);

    data_l = BIO_get_mem_data(bp, &data_p);
    obj = abcdk_object_copyfrom(data_p,data_l);        
    BIO_free(bp);

    return (obj);
#endif // #ifndef HAVE_OPENSSL
}