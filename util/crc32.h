/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_CRC32_H
#define ABCDK_UTIL_CRC32_H

#include "util/general.h"

__BEGIN_DECLS

/**
 * 计算CRC32值。
 * 
 * 宽度(32 bits)，多项式(04C11DB7)，初始值(FFFFFFFF)，结果异或值(FFFFFFFF)，输入值反转(true)，输出值反转(true)。
 * 
 * @param [int] old 上一轮的值。
 * 
*/
uint32_t abcdk_crc32_sum(const void *data,size_t size,uint32_t old);

__END_DECLS


#endif //ABCDK_UTIL_CRC32_H