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

int abcdk_openssl_pki_add_ext(X509 *cert, int nid, const char *value)
{
    X509_EXTENSION *ex_info = X509V3_EXT_conf_nid(NULL, NULL, nid, (char *)value);
    if (!ex_info)
        return -1;

    X509_add_ext(cert, ex_info, -1);
    X509_EXTENSION_free(ex_info);

    return 0;
}
