/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_OPENSSL_UTIL_H
#define ABCDK_OPENSSL_UTIL_H

#include "abcdk/util/tree.h"
#include "abcdk/util/io.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/object.h"
#include "abcdk/util/dirent.h"
#include "abcdk/util/getpass.h"
#include "abcdk/openssl/openssl.h"

__BEGIN_DECLS

/******************************************************************************************************/

/**清理全局环境.*/
void abcdk_openssl_cleanup();

/**初始化全局环境.*/
void abcdk_openssl_init();

/******************************************************************************************************/

/**释放. */
void abcdk_openssl_bn_free(BIGNUM **bn);

/**释放. */
void abcdk_openssl_evp_pkey_free(EVP_PKEY **pkey);

/**释放. */
void abcdk_openssl_evp_cipher_ctx_free(EVP_CIPHER_CTX **cipher);

/**
 * 加载密钥.
 * 
 * @note 仅支持PEM格式.
 * 
 * @param [in] key 密钥文件.
 * @param [in] pubkey 是否为公钥.!0 是, 0 否.
 * @param [out] passwd 密钥密码, NULL(0) 忽略.
 * 
*/
EVP_PKEY *abcdk_openssl_evp_pkey_load(const char *key,int pubkey, abcdk_object_t **passwd);


/******************************************************************************************************/


/**释放. */
void abcdk_openssl_rsa_free(RSA **rsa);

/**
 * 检查密钥是否为私钥.
 * 
 * @return !0 是, 0 否.
*/
int abcdk_openssl_rsa_is_prikey(RSA *rsa);

/**
 * 创建RSA密钥对象.
 * 
 * @param bits 密钥的长度(比特).
 * @param e 指数.见RSA_3/RSA_F4.
*/
RSA *abcdk_openssl_rsa_create(int bits, unsigned long e);

/**
 * 加载密钥.
 * 
 * @note 仅支持PEM格式.
 * 
 * @param [in] key 密钥文件.
 * @param [in] pubkey 是否为公钥.!0 是, 0 否.
 * @param [out] passwd 密钥密码, NULL(0) 忽略.
 * 
*/
RSA *abcdk_openssl_rsa_load(const char *key,int pubkey, abcdk_object_t **passwd);

/**
 * 导出密钥.
*/
abcdk_object_t *abcdk_openssl_rsa_export(RSA *rsa);

/**
 * 加密解密.
 * 
 * @note 加密, 输出的数据长度是“密文”的长度.
 * @note 解密, 输出的数据长度是“缓存”的长度.
*/
abcdk_object_t *abcdk_openssl_rsa_update(RSA *rsa, const uint8_t *in, int in_len, int enc);


/******************************************************************************************************/

/**
 * HMAC支持的算法.
*/
typedef enum _abcdk_openssl_hmac_type
{
    ABCDK_OPENSSL_HMAC_MD2 = 1,
#define ABCDK_OPENSSL_HMAC_MD2 ABCDK_OPENSSL_HMAC_MD2

    ABCDK_OPENSSL_HMAC_MD4 = 2,
#define ABCDK_OPENSSL_HMAC_MD4 ABCDK_OPENSSL_HMAC_MD4

    ABCDK_OPENSSL_HMAC_MD5 = 3,
#define ABCDK_OPENSSL_HMAC_MD5 ABCDK_OPENSSL_HMAC_MD5

    ABCDK_OPENSSL_HMAC_SHA = 4,
#define ABCDK_OPENSSL_HMAC_SHA ABCDK_OPENSSL_HMAC_SHA

    ABCDK_OPENSSL_HMAC_SHA1 = 5,
#define ABCDK_OPENSSL_HMAC_SHA1 ABCDK_OPENSSL_HMAC_SHA1

    ABCDK_OPENSSL_HMAC_SHA224 = 6,
#define ABCDK_OPENSSL_HMAC_SHA224 ABCDK_OPENSSL_HMAC_SHA224

    ABCDK_OPENSSL_HMAC_SHA256 = 7,
#define ABCDK_OPENSSL_HMAC_SHA256 ABCDK_OPENSSL_HMAC_SHA256

    ABCDK_OPENSSL_HMAC_SHA384 = 8,
#define ABCDK_OPENSSL_HMAC_SHA384 ABCDK_OPENSSL_HMAC_SHA384

    ABCDK_OPENSSL_HMAC_SHA512 = 9,
#define ABCDK_OPENSSL_HMAC_SHA512 ABCDK_OPENSSL_HMAC_SHA512

    ABCDK_OPENSSL_HMAC_RIPEMD160 = 10,
#define ABCDK_OPENSSL_HMAC_RIPEMD160 ABCDK_OPENSSL_HMAC_RIPEMD160

    ABCDK_OPENSSL_HMAC_WHIRLPOOL = 11
#define ABCDK_OPENSSL_HMAC_WHIRLPOOL ABCDK_OPENSSL_HMAC_WHIRLPOOL
}abcdk_openssl_hmac_type_t;

/**释放.*/
void abcdk_openssl_hmac_free(HMAC_CTX **ctx);

/**申请.*/
HMAC_CTX *abcdk_openssl_hmac_alloc();

/**
 * 初始化环境.
 * 
 *  @return 0 成功, !0 失败.
*/
int abcdk_openssl_hmac_init(HMAC_CTX *ctx,const void *key, int len,int type);


/******************************************************************************************************/


/**释放. */
void abcdk_openssl_x509_free(X509 **x509);

/** 打印证书信息.*/
abcdk_object_t *abcdk_openssl_cert_dump(X509 *x509);

/** 打印证书检验错误信息. */
abcdk_object_t *abcdk_openssl_cert_verify_error_dump(X509_STORE_CTX *store_ctx);

/** 从证书中获取RSA公钥.*/
RSA *abcdk_openssl_cert_get_rsa_pubkey(X509 *x509);

/**
 * 导出证书到内存.
 * 
 * @note 仅支持PEM格式.
*/
abcdk_object_t *abcdk_openssl_cert_to_pem(X509 *leaf_cert,STACK_OF(X509) *cert_chain);

/**
 * 加载证书吊销列表.
 * 
 * @note 仅支持PEM格式.
 * 
 * @param [in] crl 证书吊销列表.
*/
X509_CRL *abcdk_openssl_crl_load(const char *crl);

/**
 * 加载证书.
 * 
 * @note 仅支持PEM格式.
 * 
 * @param [in] cert 证书文件.
*/
X509 *abcdk_openssl_cert_load(const char *cert);



/**
 * 加载父证书.
 * 
 * @note 仅支持PEM格式.
 * 
 * @param [in] leaf_cert 叶证书.
 * @param [in] ca_path 证书目录.
 * @param [in] pattern 证书文件名称通配符, NULL(0) 忽略.
*/
X509 *abcdk_openssl_cert_father_find(X509 *leaf_cert,const char *ca_path,const char *pattern);

/**
 * 加载证书链.
 * 
 * @note 仅支持PEM格式.
 * 
 * @param [in] ca_path 证书目录.
 */
STACK_OF(X509) *abcdk_openssl_cert_chain_load(X509 *leaf_cert, const char *ca_path,const char *pattern);

/**
 * 加载证书链(内存).
 * 
 * @note 仅支持PEM格式.
 * 
 * @param [in] len 长度.-1 自动计算.
 * 
*/
STACK_OF(X509) *abcdk_openssl_cert_chain_load_mem(const char *buf,int len);

/**
 * 加载证书池.
 * 
 * @param [in] ca_file CA证书文件(内部可能包含多个证书), NULL(0) 忽略.
 * @param [in] ca_path CA证书目录, NULL(0) 忽略.
 */
X509_STORE *abcdk_openssl_cert_load_locations(const char *ca_file, const char *ca_path);

/**
 * 准备证书检验环境.
 * 
 * @param [in] leaf_cert 叶证书, NULL(0) 忽略.
 * @param [in] cert_chain 证书链, NULL(0) 忽略.
*/
X509_STORE_CTX *abcdk_openssl_cert_verify_prepare(X509_STORE *store,X509 *leaf_cert,STACK_OF(X509) *cert_chain);


/******************************************************************************************************/

/**
 * 释放SSL_CTX句柄.
*/
void abcdk_openssl_ssl_ctx_free(SSL_CTX **ctx);

/**
 * 创建SSL_CTX句柄.
 * 
 * @note 仅支持PEM格式.
 * 
 * @param server !0 服务端, 0 客户端.
 * @param cafile CA证书文件, NULL(0) 忽略.
 * @param capath CA证书目录, NULL(0) 忽略.
 * @param chk_crl 0 不检查吊销列表, 1 仅检查叶证书的吊销列表, 2 检查整个证书链路的吊销列表.
 * @param use_crt 证书.NULL(0) 忽略.
 * @param use_key 私钥.NULL(0) 忽略.
 * 
*/
SSL_CTX *abcdk_openssl_ssl_ctx_alloc(int server, const char *cafile, const char *capath, int chk_crl, X509 *use_crt, EVP_PKEY *use_key);

/**
 * 设置应用层协议和算法.
 * 
 *  @return 0 成功, -1 失败.
*/
int abcdk_openssl_ssl_ctx_set_alpn(SSL_CTX *ctx,const uint8_t *next_proto,const char *cipher_list);

/**
 * 释放SSL环境.
*/
void abcdk_openssl_ssl_free(SSL **ssl);

/**
 * 创建SSL环境.
*/
SSL *abcdk_openssl_ssl_alloc(SSL_CTX *ctx);

/**
 * 握手.
 * 
 * @param fd 文件或SOCKET句柄.
 * @param timeout 超时(毫秒).
 * 
 * @return 0 成功, -1 失败.
*/
int abcdk_openssl_ssl_handshake(int fd, SSL *ssl, int server, time_t timeout);

/**
 * 获取ALPN协议名称.
 * 
 * @return 0 成功, -1 不支持.
*/
int abcdk_openssl_ssl_get_alpn_selected(SSL *ssl,char buf[256]);


/************************************************************************************************************************/

__END_DECLS

#endif //ABCDK_OPENSSL_UTIL_H