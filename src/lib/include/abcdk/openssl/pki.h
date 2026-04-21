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
 * 导出密钥.
 * 
 * @note 仅支持PEM格式.
 * 
 * @param [in] pkey 密钥.
 * @param [in] pubkey 是否为公钥.!0 是, 0 否.
 * @param [in] passwd 密钥的密码地址, NULL(0) 忽略.
 * @param [in] passwd_len 密钥的密码长度, <= 0 忽略.
 */
abcdk_object_t *abcdk_openssl_pki_export_pkey(EVP_PKEY *pkey, int pubkey, uint8_t *passwd, int passwd_len);

/**
 * 生成序列号.
 */
ASN1_INTEGER *abcdk_openssl_pki_generate_serial(int bits);

/**
 * 字符串化序列号.
 */
abcdk_object_t *abcdk_openssl_pki_string_serial(ASN1_INTEGER *ai, int hex_or_dec);

/** 
 * 检查证书和私钥是否匹配.
 * 
 * @return 0 成功, -1 失败 , -2 不支持 , -3 其它错误. 
*/
int abcdk_openssl_pki_check_cert_and_pkey(X509 *cert,EVP_PKEY *pri_pkey);

/**
 * 生成证书.
 */
X509 *abcdk_openssl_pki_generate_cert(EVP_PKEY *pkey, ASN1_INTEGER *serial, const char *name_cn, const char *name_o, abcdk_option_t *opt,
                                      X509 *issuer_cert, EVP_PKEY *issuer_pkey);

/**
 * 导出证书.
 * 
 * @note 仅支持PEM格式.
*/
abcdk_object_t *abcdk_openssl_pki_export_cert(X509 *cert);

__END_DECLS

#endif //ABCDK_OPENSSL_PKI_H