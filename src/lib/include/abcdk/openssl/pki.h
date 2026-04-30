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


/** 销毁私钥.*/
void abcdk_openssl_pki_destroy_key(EVP_PKEY **key);

/** 生成私钥.*/
EVP_PKEY *abcdk_openssl_pki_generate_key_from_rsa(int bits);

/** 生成私钥.*/
EVP_PKEY *abcdk_openssl_pki_generate_key_from_ec(int nid);

/** 
 * 是否私钥.
 * 
 * @return !0 是, 0 否.
*/
int abcdk_openssl_pki_private_key_was(EVP_PKEY *key);

/** 私钥提取公钥.*/
EVP_PKEY *abcdk_openssl_pki_generate_key_to_public(EVP_PKEY *key);

/**
 * 导出密钥.
 * 
 * @note 仅支持PEM格式.
 * 
 * @param [in] pkey 密钥.
 * @param [in] passwd 密钥的密码地址, NULL(0) 忽略.
 * @param [in] passwd_len 密钥的密码长度, <= 0 忽略.
 */
abcdk_object_t *abcdk_openssl_pki_export_key(EVP_PKEY *key, uint8_t *passwd, int passwd_len);

/** 销毁序列号.*/
void abcdk_openssl_pki_destroy_serial(ASN1_INTEGER **serial);

/** 生成证书. */
ASN1_INTEGER *abcdk_openssl_pki_generate_serial(int bits);

/** 字符串化序列号.*/
abcdk_object_t *abcdk_openssl_pki_string_serial(ASN1_INTEGER *ai, int hex_or_dec);

/** 销毁证书.*/
void abcdk_openssl_pki_destroy_cert(X509 **cert);

/** 生成证书.*/
X509 *abcdk_openssl_pki_generate_cert(EVP_PKEY *key, ASN1_INTEGER *serial, const char *name_cn, const char *name_o, abcdk_option_t *opt,
                                      X509 *issuer_cert, EVP_PKEY *issuer_key);

/**
 * 导出证书.
 * 
 * @note 仅支持PEM格式.
*/
abcdk_object_t *abcdk_openssl_pki_export_cert(X509 *cert);

/**
 * 导入证书.
 * 
 * @note 仅支持PEM格式.
*/
X509 *abcdk_openssl_pki_import_cert(const char *data, size_t size);

/**
 * 导入证书.
 * 
 * @note 仅支持PEM格式.
*/
X509 *abcdk_openssl_pki_import_cert_from_file(const char *file);


/**
 * 导入证书.
 * 
 * @note 仅支持PEM格式.
*/
X509 *abcdk_openssl_pki_import_cert_from_fd(FILE *fp);

/** 销毁吊销列表.*/
void abcdk_openssl_pki_destroy_crl(X509_CRL **crl);

/**
 * 证书吊销检查.
 * 
 * @return 0 未吊销, !0 已吊销.
 */
int abcdk_openssl_pki_cert_is_revoked(X509_CRL *crl, X509 *cert);

/** 
 * 吊销证书.
 * 
 * @return 0 成功, -1 失败, -2 重复吊销.
*/
int abcdk_openssl_pki_revoke_cert(X509_CRL **crl, X509 *cert, int reason_code, X509 *issuer_cert, EVP_PKEY *issuer_key);

/**
 * 更新吊销列表.
 * 
 * @return 0 成功, -1 失败.
 */
int abcdk_openssl_pki_update_crl(X509_CRL *crl, abcdk_option_t *opt, X509 *issuer_cert, EVP_PKEY *issuer_key);

/**
 * 导出吊销列表.
 * 
 * @note 仅支持PEM格式.
*/
abcdk_object_t *abcdk_openssl_pki_export_crl(X509_CRL *crl);

__END_DECLS

#endif //ABCDK_OPENSSL_PKI_H