/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 * 
 */
#include "abcdk/openssl/totp.h"



uint32_t abcdk_openssl_totp_generate(const EVP_MD *evp_md_ctx, const uint8_t *key, int klen, uint64_t counter)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenSSL工具。"));
    return 0xFFFFFFFF;
#else //#ifndef HAVE_OPENSSL
    uint64_t be_counter;
    uint8_t out[128] = {0};
    uint32_t olen = 128;
    uint32_t off, otp;

    assert(evp_md_ctx != NULL && key != NULL && klen > 0);

    /*转大端字序。*/
    be_counter = abcdk_endian_h_to_b64(counter);

    HMAC(evp_md_ctx, key, klen, (uint8_t *)&be_counter, sizeof(be_counter), out, &olen);

    /*取哈希值最后一个字节的低4位作为动态偏移量。*/
    off = out[olen - 1] & 0x0F;

    /*取4字节(大端字序)，转换为32位无符号整数(本地字序)。*/
    if (abcdk_endian_check(0))
    {
        otp = ((uint32_t)(out[off + 0] & 0x7F) << 24) |
              ((uint32_t)(out[off + 1] & 0xFF) << 16) |
              ((uint32_t)(out[off + 2] & 0xFF) << 8) |
              ((uint32_t)(out[off + 3] & 0xFF));
    }
    else
    {
        otp = ((uint32_t)(out[off + 0] & 0xFF)) |
              ((uint32_t)(out[off + 1] & 0xFF) << 8) |
              ((uint32_t)(out[off + 2] & 0xFF) << 16) |
              ((uint32_t)(out[off + 3] & 0x7F) << 24);
    }

    return otp;
#endif //#ifndef HAVE_OPENSSL
}

uint32_t abcdk_openssl_totp_generate_sha1(const uint8_t *key, int klen, uint64_t counter)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenSSL工具。"));
    return 0xFFFFFFFF;
#else //#ifndef HAVE_OPENSSL
    assert(key != NULL && klen > 0);

    return abcdk_openssl_totp_generate(EVP_sha1(),key,klen,counter);
#endif //#ifndef HAVE_OPENSSL
}


uint32_t abcdk_openssl_totp_generate_sha256(const uint8_t *key, int klen, uint64_t counter)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenSSL工具。"));
    return 0xFFFFFFFF;
#else //#ifndef HAVE_OPENSSL
    assert(key != NULL && klen > 0);

    return abcdk_openssl_totp_generate(EVP_sha256(),key,klen,counter);
#endif //#ifndef HAVE_OPENSSL
}

uint32_t abcdk_openssl_totp_generate_sha512(const uint8_t *key, int klen, uint64_t counter)
{
#ifndef HAVE_OPENSSL
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenSSL工具。"));
    return 0xFFFFFFFF;
#else //#ifndef HAVE_OPENSSL
    assert(key != NULL && klen > 0);

    return abcdk_openssl_totp_generate(EVP_sha512(),key,klen,counter);
#endif //#ifndef HAVE_OPENSSL
}
