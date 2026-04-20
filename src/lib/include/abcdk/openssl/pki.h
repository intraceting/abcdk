/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2026 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_OPENSSL_PKI_H
#define ABCDK_OPENSSL_PKI_H

#include "abcdk/util/option.h"
#include "abcdk/openssl/openssl.h"

__BEGIN_DECLS


/**
 * 生成私钥.
 */
EVP_PKEY *abcdk_openssl_pki_generate_pkey(int bits);

/**
 * 生成序列号.
 */
ASN1_INTEGER *abcdk_openssl_pki_generate_serial(int bits);

/** 
 * 检查证书和私钥是否匹配.
 * 
 * @return 0 成功, -1 失败 , -2 不支持 , -3 其它错误. 
*/
int abcdk_openssl_pki_check_cert_and_pkey(X509 *cert,EVP_PKEY *pri_pkey);

/**
 * 生成证书.
 */
X509 *abcdk_openssl_pki_issue_cert(EVP_PKEY *pkey, ASN1_INTEGER *serial, const char *cn, const char *org, int ca_or_not, abcdk_option_t *opt,
                                   X509 *issuer_cert, EVP_PKEY *issuer_pkey);

__END_DECLS

#endif //ABCDK_OPENSSL_PKI_H