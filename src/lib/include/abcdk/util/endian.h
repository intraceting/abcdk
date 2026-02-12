/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_ENDIAN_H
#define ABCDK_UTIL_ENDIAN_H

#include "abcdk/util/defs.h"

__BEGIN_DECLS

/**
 * 字节序检测
 * 
 * @param big 0 检测是否为小端字节序, !0 检测是否为大端字节序.
 * 
 * @return 0 否, !0 是.
 */
int abcdk_endian_check(int big);

/**
 * 字节序交换.
 * 
 * @return dst
*/
uint8_t* abcdk_endian_swap(uint8_t* dst,int len);

/**
 * 大端字节序转本地字节序.
 * 
 * @note 如果本地是大端字节序, 会忽略.
*/
uint8_t* abcdk_endian_b_to_h(uint8_t* dst,int len);

/**
 * 16位整型数值, 大端字节序转本地字节序.
*/
uint16_t abcdk_endian_b_to_h16(uint16_t src);

/**
 * 24位整型数值, 大端字节序转本地字节序.
*/
uint32_t abcdk_endian_b_to_h24(const uint8_t* src);

/**
 * 32位整型数值, 大端字节序转本地字节序.
*/
uint32_t abcdk_endian_b_to_h32(uint32_t src);

/**
 * 64位整型数值, 大端字节序转本地字节序.
*/
uint64_t abcdk_endian_b_to_h64(uint64_t src);

/**
 * 本地字节序转大端字节序.
 * 
 * @note 如果本地是大端字节序, 会忽略.
*/
uint8_t* abcdk_endian_h_to_b(uint8_t* dst,int len);

/**
 * 16位整型数值, 本地字节序转大端字节序.
*/
uint16_t abcdk_endian_h_to_b16(uint16_t src);

/**
 * 24位整型数值, 本地字节序转大端字节序.
*/
uint8_t* abcdk_endian_h_to_b24(uint8_t* dst,uint32_t src);

/**
 * 32位整型数值, 本地字节序转大端字节序.
*/
uint32_t abcdk_endian_h_to_b32(uint32_t src);

/**
 * 64位整型数值, 本地字节序转大端字节序.
*/
uint64_t abcdk_endian_h_to_b64(uint64_t src);

/**
 * 小端字节序转本地字节序.
 * 
 * @note 如果本地是小端字节序, 会忽略.
*/
uint8_t* abcdk_endian_l_to_h(uint8_t* dst,int len);

/**
 * 16位整型数值, 小端字节序转本地字节序.
*/
uint16_t abcdk_endian_l_to_h16(uint16_t src);

/**
 * 24位整型数值, 小端字节序转本地字节序.
*/
uint32_t abcdk_endian_l_to_h24(uint8_t* src);

/**
 * 32位整型数值, 小端字节序转本地字节序.
*/
uint32_t abcdk_endian_l_to_h32(uint32_t src);

/**
 * 64位整型数值, 小端字节序转本地字节序.
*/
uint64_t abcdk_endian_l_to_h64(uint64_t src);

/**
 * 本地字节序转小端字节序.
 * 
 * @note 如果本地是小端字节序, 会忽略.
*/
uint8_t* abcdk_endian_h_to_l(uint8_t* dst,int len);

/**
 * 16位整型数值, 本地字节序转小端字节序.
*/
uint16_t abcdk_endian_h_to_l16(uint16_t src);

/**
 * 24位整型数值, 本地字节序转小端字节序.
*/
uint8_t* abcdk_endian_h_to_l24(uint8_t* dst,uint32_t src);

/**
 * 32位整型数值, 本地字节序转小端字节序.
*/
uint32_t abcdk_endian_h_to_l32(uint32_t src);

/**
 * 64位整型数值, 本地字节序转小端字节序.
*/
uint64_t abcdk_endian_h_to_l64(uint64_t src);

__END_DECLS

#endif //ABCDK_UTIL_ENDIAN_H