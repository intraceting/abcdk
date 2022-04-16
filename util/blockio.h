/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_UTIL_BLOCKIO_H
#define ABCDK_UTIL_BLOCKIO_H

#include "util/general.h"
#include "util/buffer.h"

__BEGIN_DECLS

/**
 * 以块为单位读数据。
 * 
 * @param buf 缓存。NULL(0) 自由块大小，!NULL(0) 定长块大小。
 * 
 * @return > 0 读取的长度，<= 0 读取失败或已到末尾。
*/
ssize_t abcdk_block_read(int fd, void *data, size_t size,abcdk_buffer_t *buf);

/**
 * 以块为单位写数据。
 * 
 * @param buf 缓存。NULL(0) 自由块大小，!NULL(0) 定长块大小。
 * 
 * @return > 0 写入的长度，<= 0 写入失败或空间不足。
*/
ssize_t abcdk_block_write(int fd, const void *data, size_t size,abcdk_buffer_t *buf);

/**
 * 以块为单位写补齐数据。
 * 
 * @param stuffing 填充物。
 * @param buf 缓存。NULL(0) 自由块大小，!NULL(0) 定长块大小。
 * 
 * @return > 0 缓存数据全部写完，= 0 缓存无数据或无缓存，< 0 写入失败或空间不足(剩余数据在缓存中)。
*/
int abcdk_block_write_trailer(int fd, uint8_t stuffing,abcdk_buffer_t *buf);

__END_DECLS


#endif //ABCDK_UTIL_BLOCKIO_H
