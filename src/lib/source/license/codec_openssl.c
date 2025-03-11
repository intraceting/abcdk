/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/license/codec.h"


#ifdef OPENSSL_VERSION_NUMBER

abcdk_object_t *abcdk_license_codec_encrypt_openssl(const void *src, size_t slen, int enc, void *opaque)
{
    abcdk_openssl_cipher_t *ctx;
    abcdk_object_t *dst = NULL;
    int chk;
    
    ctx = abcdk_openssl_cipher_create(ABCDK_OPENSSL_CIPHER_SCHEME_AES256GCM,(uint8_t*)opaque,strlen(opaque));
    if(!ctx)
        return NULL;

    dst = abcdk_openssl_cipher_update_pack(ctx,(uint8_t*)src,slen,enc);
    if(!dst)
        goto ERR;

    abcdk_openssl_cipher_destroy(&ctx);
    return dst;

ERR:

    abcdk_openssl_cipher_destroy(&ctx);
    abcdk_object_unref(&dst);
    return NULL;
}

#else //OPENSSL_VERSION_NUMBER

abcdk_object_t *abcdk_license_codec_encrypt_openssl(const void *src, size_t slen, int enc, void *opaque)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenSSL工具。"));
    return NULL;
}

#endif //OPENSSL_VERSION_NUMBER