/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_LICENSE_CODEC_H
#define ABCDK_LICENSE_CODEC_H

#include "abcdk/license/license.h"

__BEGIN_DECLS

/**
 * 加(解)密函数。
*/
typedef abcdk_object_t *(*abcdk_license_codec_encrypt_cb)(const void *src, size_t slen, int enc, void *opaque);

/**
 * 编码。
 *
 * @return 0 成功，-1 失败(无效的或错误的)。
 */
int abcdk_license_codec_encode(abcdk_object_t **dst, const abcdk_license_info_t *src, abcdk_license_codec_encrypt_cb encrypt_cb, void *opaque);

/**
 * 解码。
 *
 * @return 0 成功，-1 失败(无效的或错误的)。
 */
int abcdk_license_codec_decode(abcdk_license_info_t *dst, const char *src, abcdk_license_codec_encrypt_cb encrypt_cb, void *opaque);

/**
 * 加(解)密函数。
*/
abcdk_object_t *abcdk_license_codec_encrypt_openssl(const void *src, size_t slen, int enc, void *opaque);


__END_DECLS

#endif // ABCDK_LICENSE_CODEC_H
