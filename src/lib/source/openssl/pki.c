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
#include "abcdk/openssl/openssl.h"
#include "abcdk/openssl/pki.h"
#include "abcdk/openssl/util.h"

EVP_PKEY *abcdk_openssl_pki_generate_pkey(int bits)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else //#ifndef HAVE_OPENSSL 
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
#endif //#ifndef HAVE_OPENSSL
}

ASN1_INTEGER *abcdk_openssl_pki_generate_serial(int bits)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else //#ifndef HAVE_OPENSSL 
    BIGNUM *bn = NULL;
    ASN1_INTEGER *serial = NULL;

    assert(bits > 0 && (bits % 2) == 0);

    bn = BN_new();
    if(!bn)
        return NULL;

    /*
     * 随机生成. 
     * RFC要求: 最高位不能全是1, 避免负数.
    */
    BN_rand(bn, bits, BN_RAND_TOP_ANY, BN_RAND_BOTTOM_ANY);

    serial = BN_to_ASN1_INTEGER(bn, NULL);
    abcdk_openssl_bn_free(&bn);
  
    return serial;
#endif //#ifndef HAVE_OPENSSL
}

#ifdef HAVE_OPENSSL
static int _abcdk_openssl_pki_add_ext(X509 *cert, int nid, const char *value,...)
{
    char buf[8000] = {0};

    va_list ap;
    va_start(ap, value);
    vsnprintf(buf, 8000, value, ap);
    va_end(ap);

    X509_EXTENSION *ex_info = X509V3_EXT_conf_nid(NULL, NULL, nid, buf);
    if (!ex_info)
        return -1;

    X509_add_ext(cert, ex_info, -1);
    X509_EXTENSION_free(ex_info);

    return 0;
}
#endif //#ifdef HAVE_OPENSSL

X509 *abcdk_openssl_pki_issue_cert(EVP_PKEY *pkey, ASN1_INTEGER *serial, const char *cn, const char *org, int ca_or_not, abcdk_option_t *opt,
                                   X509 *issuer_cert, EVP_PKEY *issuer_pkey)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("当前环境在构建时未包含OPENSSL工具."));
    return NULL;
#else // #ifndef HAVE_OPENSSL
    X509 *cert = NULL;

    assert(serial != NULL && cn != NULL && org != NULL && opt != NULL);
    assert(*cn != '\0' && *org != '\0');

    cert = X509_new();
    if(!cert)
        return NULL;

    long not_before_days = abcdk_option_get_long(opt,"--not-before-days",0,0);
    long not_after_days = abcdk_option_get_long(opt,"--not-before-days",0,30);
    long pathlen = abcdk_option_get_long(opt,"--pathlen",0,0);
    const char *san = abcdk_option_get(opt,"--subject-alt-name",0,NULL);

    X509_set_version(cert, 2);
    X509_set_serialNumber(cert, serial);
    X509_gmtime_adj(X509_get_notBefore(cert), not_before_days * 24 * 3600);
    X509_gmtime_adj(X509_get_notAfter(cert), not_after_days * 24 * 3600);

    X509_set_pubkey(cert, pkey);

    // Subject
    X509_NAME *name = X509_get_subject_name(cert);

    X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC, (unsigned char*)cn, -1, -1, 0);
    X509_NAME_add_entry_by_txt(name, "O", MBSTRING_ASC, (unsigned char*)org, -1, -1, 0);

    // Issuer
    if (issuer_cert)
        X509_set_issuer_name(cert, X509_get_subject_name(issuer_cert));
    else
        X509_set_issuer_name(cert, name);

    if (ca_or_not)
    {
        _abcdk_openssl_pki_add_ext(cert, NID_basic_constraints, "critical,CA:TRUE,pathlen:%d", pathlen);
        _abcdk_openssl_pki_add_ext(cert, NID_key_usage, "critical,keyCertSign,cRLSign");
    }
    else
    {
        _abcdk_openssl_pki_add_ext(cert, NID_basic_constraints, "CA:FALSE");
        _abcdk_openssl_pki_add_ext(cert, NID_key_usage, "digitalSignature,keyEncipherment");
        _abcdk_openssl_pki_add_ext(cert, NID_ext_key_usage, "serverAuth,clientAuth");

        if (san)
            _abcdk_openssl_pki_add_ext(cert, NID_subject_alt_name, san);
    }


#endif // #ifndef HAVE_OPENSSL
}
