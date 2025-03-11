/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_RANDOM_H
#define ABCDK_UTIL_RANDOM_H

#include "abcdk/util/defs.h"
#include "abcdk/util/atomic.h"
#include "abcdk/util/io.h"

__BEGIN_DECLS

/**
 * 获取一个随机数。
 * 
 * @warning 非常慢，能不用则不用。
 * 
 * @param [in out] seed 随机种子。
 */
uint64_t abcdk_rand(uint64_t min, uint64_t max);

/**
 * 产生随机字符。
 * 
 * @param [in] type 类型。0 所有可见字符，1 所有字母和数字，2 所在大写字母，3 所有小字字母，4 所有数字，5 所有字符。
*/
uint8_t *abcdk_rand_bytes(uint8_t *buf,size_t size,int type);

/** 
 * 洗牌算法元素交换回调函数。
*/
typedef void (*abcdk_rand_shuffle_swap_cb)(size_t a,size_t b, void *opaque);

/**
 * 洗牌算法。
 * 
 * @note Fisher-Yates
 * 
 * @param [in out] seed 随机种子。
 * @param [in] size 元素数量。
*/
void abcdk_rand_shuffle(size_t size,abcdk_rand_shuffle_swap_cb swap_cb,void *opaque);

/**
 * 数组洗牌。
 * 
 * @param [in out] buf 数组首地址。
 * @param [in] count 数组元素数量。
 * @param [in out] seed 随机种子。
 * @param [in] type 元素类型。1 uint8(int8)，2 uint16(int16)，3 uint32(int32)，4 uint64(int64)，5 float，6 double。
 * 
*/
void *abcdk_rand_shuffle_array(void *buf,size_t count,int type);


__END_DECLS

#endif //ABCDK_UTIL_RANDOM_H