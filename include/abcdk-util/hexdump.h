/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_HEXDUMP_H
#define ABCDK_UTIL_HEXDUMP_H

#include "general.h"
#include "allocator.h"

/**
 * 打印16进制格式。
 * 
 * @return > 0 成功(打印的总长度)，<=0 失败(空间不足或出错)。
*/
ssize_t abcdk_hexdump(FILE *fd, const void *data, size_t size,
                      abcdk_allocator_t *keywords, abcdk_allocator_t *palette);

/**
 * 打印16进制格式。
 * 
 * @return > 0 成功(打印的总长度)，<=0 失败(空间不足或出错)。
*/
ssize_t abcdk_hexdump2(const char *file, const void *data, size_t size,
                       abcdk_allocator_t *keywords, abcdk_allocator_t *palette);

/*------------------------------------------------------------------------------------------------*/


#endif //
