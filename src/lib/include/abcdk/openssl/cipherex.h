/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_OPENSSL_CIPHEREX_H
#define ABCDK_OPENSSL_CIPHEREX_H

#include "abcdk/openssl/cipher.h"

__BEGIN_DECLS

/**简单的加密接口. */
typedef struct _abcdk_openssl_cipherex abcdk_openssl_cipherex_t;

/**销毁.*/
void abcdk_openssl_cipherex_destroy(abcdk_openssl_cipherex_t **ctx);

/**
 * 创建.
 * 
 * @param [in] worker 工人数量.
*/
abcdk_openssl_cipherex_t *abcdk_openssl_cipherex_create(int worker, int scheme, const uint8_t *key, size_t klen);

/**执行.*/
abcdk_object_t *abcdk_openssl_cipherex_update(abcdk_openssl_cipherex_t *ctx, const uint8_t *in, int in_len, int enc);

/**执行.*/
abcdk_object_t *abcdk_openssl_cipherex_update_pack(abcdk_openssl_cipherex_t *ctx, const uint8_t *in, int in_len, int enc);


__END_DECLS

#endif //ABCDK_OPENSSL_CIPHEREX_H