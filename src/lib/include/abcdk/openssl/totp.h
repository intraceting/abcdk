/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_OPENSSL_TOTP_H
#define ABCDK_OPENSSL_TOTP_H

#include "abcdk/util/endian.h"
#include "abcdk/openssl/util.h"

__BEGIN_DECLS

/**
 * 基于时间的一次性密码。
 *
 * @note RFC6238
 */
uint32_t abcdk_openssl_totp_generate(const EVP_MD *evp_md_ctx, const uint8_t *key, int klen, uint64_t counter);

/**
 * 基于时间的一次性密码。
 *
 * @note RFC6238(sha-1,sha-128）
 */
uint32_t abcdk_openssl_totp_generate_sha1(const uint8_t *key, int klen, uint64_t counter);

/**
 * 基于时间的一次性密码。
 *
 * @note RFC6238(sha-256）
 */
uint32_t abcdk_openssl_totp_generate_sha256(const uint8_t *key, int klen, uint64_t counter);

/**
 * 基于时间的一次性密码。
 *
 * @note RFC6238(sha-512）
 */
uint32_t abcdk_openssl_totp_generate_sha512(const uint8_t *key, int klen, uint64_t counter);

__END_DECLS


#endif //ABCDK_OPENSSL_TOTP_H