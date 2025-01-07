/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_BLOOM_H
#define ABCDK_UTIL_BLOOM_H

#include "abcdk/util/defs.h"
#include "abcdk/util/endian.h"

__BEGIN_DECLS

/** 
 * 布隆-插旗
 * 
 * @note 以字节的二进制阅读顺序排列。如：0(7)~7(0) 8(7)~15(0) 16(7)~23(0) 24(7)~31(0) ... 。
 * 
 * @param size 池大小(Bytes)
 * @param index 索引(Bits)。有效范围：0 ～ size*8-1。
 * 
 * @return 0 成功，1 成功（或重复操作）。
*/
int abcdk_bloom_mark(uint8_t *pool, size_t size, size_t index);

/** 
 * 布隆-拔旗
 * 
 * @return 0 成功，1 成功（或重复操作）。
*/
int abcdk_bloom_unset(uint8_t* pool,size_t size,size_t index);

/**
 * 布隆-过滤
 * 
 * @return 0 不存在，1 已存在。
*/
int abcdk_bloom_filter(const uint8_t* pool,size_t size,size_t index);

/**
 * 布隆-写 
 * 
 * @param offset 偏移量(Bits)。有效范围：0 ～ size*8-1。
 * @param val 值。0 (0)，1 (!0)。
*/
void abcdk_bloom_write(uint8_t* pool,size_t size,size_t offset,int val);

/**
 * 布隆-读
 * 
 * @param offset 偏移量(Bits)。有效范围：0 ～ size*8-1。
 * 
 * @return 0 或 !0。
 */
int abcdk_bloom_read(const uint8_t* pool,size_t size,size_t offset);

/**
 * 布隆-读转数值。
*/
uint64_t abcdk_bloom_read_number(const uint8_t *pool, size_t size, size_t offset, int bits);

/**
 * 布隆-数值转写。
 */
void abcdk_bloom_write_number(uint8_t *pool, size_t size, size_t offset, int bits, uint64_t num);

__END_DECLS

#endif //ABCDK_UTIL_BLOOM_H