/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_BASECODE_H
#define ABCDK_UTIL_BASECODE_H

#include "abcdk/util/general.h"
#include "abcdk/util/math.h"
#include "abcdk/util/bloom.h"
#include "abcdk/util/object.h"


__BEGIN_DECLS

/** 
 * 简单的base编/解码器。
*/
typedef struct _abcdk_basecode
{
    /**
     * 基数。
    */
    uint8_t base;

    /**
     * 编码表(回调)。
    */
    uint8_t (*encode_table_cb)(uint8_t n);

    /** 
     * 解码表(回调)。
    */
    uint8_t (*decode_table_cb)(uint8_t c);

    /** 补齐字符。*/
    uint8_t pad;

}abcdk_basecode_t;

/**
 * 初始化。
*/
void abcdk_basecode_init(abcdk_basecode_t *ctx, uint8_t base);

/**
 * 编码。
 * 
 * @param [in] src 原文(指针)。
 * @param [out] dst 密文(指针)。
 * 
 * @return 原文编码后的长度。
*/
ssize_t abcdk_basecode_encode(const abcdk_basecode_t *ctx,
                              const uint8_t *src, size_t slen,
                              uint8_t *dst, size_t dmaxlen);

/**
 * 解码。
 * 
 * @param [in] src 密文(指针)。
 * @param [out] dst 原文(指针)。
 * 
 * @return 密文解码后的长度。
*/
ssize_t abcdk_basecode_decode(const abcdk_basecode_t *ctx,
                              const uint8_t *src, size_t slen,
                              uint8_t *dst, size_t dmaxlen);

/**
 * 编码。
*/
abcdk_object_t *abcdk_basecode_encode2(const void *src,size_t len, uint8_t base);

/**
 * 解码。
*/
abcdk_object_t *abcdk_basecode_decode2(const char *src,size_t len, uint8_t base);

__END_DECLS

#endif //ABCDK_UTIL_BASECODE_H