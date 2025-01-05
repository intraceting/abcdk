/*
 * This file is part of ABCDK.
 *
 * Copyright (c) intraceting<intraceting@outlook.com>
 *
 */
#ifndef ABCDK_OPENSSL_CIPHER_H
#define ABCDK_OPENSSL_CIPHER_H

#include "abcdk/util/bloom.h"
#include "abcdk/util/sha256.h"
#include "abcdk/util/crc.h"
#include "abcdk/util/spinlock.h"
#include "abcdk/openssl/openssl.h"

__BEGIN_DECLS

#ifdef OPENSSL_VERSION_NUMBER

/**简单的加密接口。 */
typedef struct _abcdk_openssl_cipher abcdk_openssl_cipher_t;

/**方案。*/
typedef enum _abcdk_openssl_cipher_scheme
{
    /*AES-256-GCM。*/
    ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM = 1,
#define ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM

    /*AES-256-CBC。*/
    ABCDK_OPENSSL_CIPHER_SCHEME_AES256CBC = 2,
#define ABCDK_OPENSSL_CIPHER_SCHEME_AES256CBC ABCDK_OPENSSL_CIPHER_SCHEME_AES256CBC
} abcdk_openssl_cipher_scheme_t;

/**销毁。*/
void abcdk_openssl_cipher_destroy(abcdk_openssl_cipher_t **ctx);

/**创建。*/
abcdk_openssl_cipher_t *abcdk_openssl_cipher_create(int scheme, const uint8_t *key, size_t key_len);

/**创建。*/
abcdk_openssl_cipher_t *abcdk_openssl_cipher_create_from_file(int scheme, const char *key_file);

/**
 * 执行。
 * 
 * @note 加密，输出的数据长度是“密文”的长度。
 * @note 解密，输出的数据长度是“缓存”的长度。
*/
abcdk_object_t *abcdk_openssl_cipher_update(abcdk_openssl_cipher_t *ctx, const uint8_t *in, int in_len, int enc);

/**
 * 执行。
 * 
 * @note 加密时，明文长度不能超过16777215(16MB-1)字节。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败(内存不足或解密错误)。
*/
abcdk_object_t *abcdk_openssl_cipher_update_pack(abcdk_openssl_cipher_t *ctx, const uint8_t *in, int in_len, int enc);

/**加锁。 */
void abcdk_openssl_cipher_lock(abcdk_openssl_cipher_t *ctx);

/**解锁。 */
int abcdk_openssl_cipher_unlock(abcdk_openssl_cipher_t *ctx,int exitcode);

#endif // OPENSSL_VERSION_NUMBER

__END_DECLS

#endif // ABCDK_OPENSSL_CIPHER_H