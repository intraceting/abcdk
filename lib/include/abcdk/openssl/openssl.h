/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_OPENSSL_OPENSSL_H
#define ABCDK_OPENSSL_OPENSSL_H

#include "abcdk/util/general.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/io.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/object.h"
#include "abcdk/util/dirent.h"

#ifdef HAVE_OPENSSL
#include <openssl/opensslconf.h>
#include <openssl/opensslv.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/ssl.h>
#include <openssl/rand.h>
#include <openssl/evp.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/x509_vfy.h>

#if !defined(OPENSSL_NO_SHA) && (!defined(OPENSSL_NO_SHA0) || !defined(OPENSSL_NO_SHA1))
#include <openssl/sha.h>
#endif //!defined(OPENSSL_NO_SHA) && (!defined(OPENSSL_NO_SHA0) || !defined(OPENSSL_NO_SHA1))

#ifndef OPENSSL_NO_RSA
#include <openssl/rsa.h>
#endif //OPENSSL_NO_RSA

#ifndef OPENSSL_NO_AES
#include <openssl/aes.h>
#endif //OPENSSL_NO_AES

#ifndef OPENSSL_NO_HMAC
#include <openssl/hmac.h>
#endif //OPENSSL_NO_HMAC

#endif //HAVE_OPENSSL

__BEGIN_DECLS

/******************************************************************************************************/

/**清理全局环境。*/
void abcdk_openssl_cleanup();

/**初始化全局环境。*/
void abcdk_openssl_init();

/******************************************************************************************************/

#ifdef HEADER_RSA_H

/**检查密钥是否为私钥。 */
int abcdk_openssl_rsa_is_private_key(RSA *rsa);

/**
 * 创建RSA密钥对象。
 * 
 * @param bits KEY的长度(bits)
 * @param e 指数，见RSA_3/RSA_F4
*/
RSA *abcdk_openssl_rsa_create(int bits, unsigned long e);

/**
 * 导出密钥。
*/
abcdk_object_t *abcdk_openssl_rsa_export(RSA *key);



#endif //EADER_RSA_H

/******************************************************************************************************/

#ifdef HEADER_HMAC_H

/**
 * HMAC支持的算法。
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

/**
 * 初始化环境。
 * 
 *  @return 0 成功，!0 失败。
*/
int abcdk_openssl_hmac_init(HMAC_CTX *hmac,const void *key, int len,int type);


#endif //HEADER_HMAC_H

/******************************************************************************************************/


#ifdef HEADER_X509_H

/** 打印证书信息。*/
abcdk_object_t *abcdk_openssl_cert_dump(X509 *x509);

/** 打印证书检验错误信息。 */
abcdk_object_t *abcdk_openssl_cert_verify_error_dump(X509_STORE_CTX *store_ctx);

/** 从证书中获取公钥。*/
RSA *abcdk_openssl_cert_pubkey(X509 *x509);


/**
 * 导出证书到内存。
 * 
 * @note 仅支持PEM格式。
*/
abcdk_object_t *abcdk_openssl_cert_to_pem(X509 *leaf_cert,STACK_OF(X509) *cert_chain);

/**
 * 加载证书吊销列表。
 * 
 * @note 仅支持PEM格式。
 * 
 * @param crl 证书吊销列表。
 * @param pwd 密码，NULL(0) 忽略。
*/
X509_CRL *abcdk_openssl_cert_crl_load(const char *crl, const char *pwd);

/**
 * 加载证书。
 * 
 * @note 仅支持PEM格式。
 * 
 * @param cert 证书文件。
 * @param pwd 密码，NULL(0) 忽略。
*/
X509 *abcdk_openssl_cert_load(const char *cert, const char *pwd);

/**
 * 加载父证书。
 * 
 * @note 仅支持PEM格式。
 * 
 * @param leaf_cert 叶证书。
 * @param ca_path 证书目录。
 * @param pattern 证书文件名称通配符，NULL(0) 忽略。
*/
X509 *abcdk_openssl_cert_father_find(X509 *leaf_cert,const char *ca_path,const char *pattern);

/**
 * 加载证书链。
 * 
 * @note 仅支持PEM格式。
 * 
 * @param ca_path 证书目录。
 */
STACK_OF(X509) *abcdk_openssl_cert_chain_load(X509 *leaf_cert, const char *ca_path,const char *pattern);

/**
 * 加载证书链(内存)。
 * 
 * @note 仅支持PEM格式。
 * 
 * @param len 长度。-1 自动计算。
 * 
*/
STACK_OF(X509) *abcdk_openssl_cert_chain_load_mem(const char *buf,int len);

/**
 * 加载证书池。
 * 
 * @param ca_file CA证书文件(内部可能包含多个证书)，NULL(0) 忽略。
 * @param ca_path CA证书目录，NULL(0) 忽略。
 */
X509_STORE *abcdk_openssl_cert_load_locations(const char *ca_file, const char *ca_path);

/**
 * 准备证书检验环境。
 * 
 * @param leaf_cert 叶证书，NULL(0) 忽略。
 * @param cert_chain 证书链，NULL(0) 忽略。
*/
X509_STORE_CTX *abcdk_openssl_cert_verify_prepare(X509_STORE *store,X509 *leaf_cert,STACK_OF(X509) *cert_chain);


#endif //HEADER_X509_H

/******************************************************************************************************/

#ifdef HEADER_SSL_H

/**
 * 释放SSL_CTX句柄。
*/
void abcdk_openssl_ssl_ctx_free(SSL_CTX **ctx);

/**
 * 创建SSL_CTX句柄。
 * 
 * @param server !0 服务端环境，0 客户端环境。
 * @param cafile CA证书文件的指针，NULL(0) 忽略。仅支持PEM格式。
 * @param capath CA证书目录的指针，NULL(0) 忽略。仅支持PEM格式。
 * @param crl_check 0 不检查吊销列表，1 仅检查叶证书的吊销列表，2 检查整个证书链路的吊销列表。
 * 
*/
SSL_CTX *abcdk_openssl_ssl_ctx_alloc(int server,const char *cafile,const char *capath,int crl_check);

/**
 * SSL_CTX加载证书、私钥。
 * 
 * @param crt 证书文件的指针。仅支持PEM格式。
 * @param key 私钥文件的指针，NULL(0) 忽略。仅支持PEM格式。
 * @param pwd 密码的指针，NULL(0) 忽略。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_openssl_ssl_ctx_load_crt(SSL_CTX *ctx,const char *crt,const char *key,const char *pwd);

/**
 * 创建SSL_CTX环境，并加载证书和私钥。
 * 
 * @note 当指定CA证书时，将检查对端证书。
 * @note 当指定CA路径时，将检查吊销列表和对端证书。
*/
SSL_CTX *abcdk_openssl_ssl_ctx_alloc_load(int server,const char *cafile,const char *capath,const char *crt,const char *key,const char *pwd);

/**
 * 释放SSL环境。
*/
void abcdk_openssl_ssl_free(SSL **ssl);

/**
 * 创建SSL环境。
*/
SSL *abcdk_openssl_ssl_alloc(SSL_CTX *ctx);

/**
 * 握手。
 * 
 * @param fd 文件或SOCKET句柄。
 * @param timeout 超时(毫秒)。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_openssl_ssl_handshake(int fd, SSL *ssl, int server, time_t timeout);

/**
 * 获取ALPN协议名称。
 * 
 * @return 0 成功，-1 不支持。
*/
int abcdk_openssl_ssl_get_alpn_selected(SSL *ssl,char buf[256]);

#endif //HEADER_SSL_H

/************************************************************************************************************************/

__END_DECLS

#endif //ABCDK_OPENSSL_OPENSSL_H