/*
 * This file is part of ABCDK.
 *
 * MIT License
 *
 */
#ifndef ABCDK_OPENSSL_CIPHER_H
#define ABCDK_OPENSSL_CIPHER_H

#include "abcdk/util/bloom.h"
#include "abcdk/util/sha256.h"
#include "abcdk/openssl/openssl.h"

__BEGIN_DECLS

#ifdef OPENSSL_VERSION_NUMBER

/**简单的加密接口。 */
typedef struct _abcdk_cipher abcdk_cipher_t;

/**
 * 方案。
 */
typedef enum _abcdk_cipher_scheme
{
    /*RSA私钥。*/
    ABCDK_CIPHER_SCHEME_RSA_PRIVATE = 1,
#define ABCDK_CIPHER_SCHEME_RSA_PRIVATE ABCDK_CIPHER_SCHEME_RSA_PRIVATE

    /*RSA公钥。*/
    ABCDK_CIPHER_SCHEME_RSA_PUBLIC = 2,
#define ABCDK_CIPHER_SCHEME_RSA_PUBLIC ABCDK_CIPHER_SCHEME_RSA_PUBLIC

    /*AES-256-GCM。*/
    ABCDK_CIPHER_SCHEME_AES_256_GCM = 3,
#define ABCDK_CIPHER_SCHEME_AES_256_GCM ABCDK_CIPHER_SCHEME_AES_256_GCM

    /*AES-256-CBC。*/
    ABCDK_CIPHER_SCHEME_AES_256_CBC = 4,
#define ABCDK_CIPHER_SCHEME_AES_256_CBC ABCDK_CIPHER_SCHEME_AES_256_CBC
} abcdk_cipher_scheme_t;

/**销毁。*/
void abcdk_cipher_destroy(abcdk_cipher_t **ctx);

/**创建。*/
abcdk_cipher_t *abcdk_cipher_create(int scheme, const uint8_t *key, size_t key_len);

/**创建。*/
abcdk_cipher_t *abcdk_cipher_create_from_file(int scheme, const char *key_file);

/**
 * 执行。
 * 
 * @note 加密，输出的数据长度是“密文”的长度。
 * @note 解密，输出的数据长度是“缓存”的长度。
*/
abcdk_object_t *abcdk_cipher_update(abcdk_cipher_t *ctx, const uint8_t *in, int in_len, int enc);

#endif // OPENSSL_VERSION_NUMBER

__END_DECLS

#endif // ABCDK_OPENSSL_CIPHER_H