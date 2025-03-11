/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_MP4_FILE_H
#define ABCDK_MP4_FILE_H

#include "abcdk/util/general.h"
#include "abcdk/util/endian.h"
#include "abcdk/util/io.h"

__BEGIN_DECLS

/**
 * 获取文件大小。
 * 
 * @note 不会影响文件指针位置。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_mp4_size(int fd, uint64_t *size);

/**
 * MP4读数据。
 * 
 * @return 0 成功，-1 失败(数据不足)，-2 失败(已到末尾或出错)。
*/
int abcdk_mp4_read(int fd, void *data, size_t size);

/**
 * MP4读数据(1字节整型转4字节)。
 * 
 * @note 自动转换为本地字节序。
*/
int abcdk_mp4_read_u8to32(int fd, uint32_t *data);

/**
 * MP4读数据(2字节整型)。
 * 
 * @note 自动转换为本地字节序。
*/
int abcdk_mp4_read_u16(int fd, uint16_t *data);

/**
 * MP4读数据(2字节整型转4字节)。
 * 
 * @note 自动转换为本地字节序。
*/
int abcdk_mp4_read_u16to32(int fd, uint32_t *data);

/**
 * MP4读数据(3字节整型)。
 * 
 * @note 自动转换为本地字节序。
*/
int abcdk_mp4_read_u24(int fd, uint8_t *data);

/**
 * MP4读数据(3字节整型转4字节)。
 * 
 * @note 自动转换为本地字节序。
*/
int abcdk_mp4_read_u24to32(int fd, uint32_t *data);

/**
 * MP4读数据(4字节整型)。
 * 
 * @note 自动转换为本地字节序。
*/
int abcdk_mp4_read_u32(int fd, uint32_t *data);

/**
 * MP4读数据(4字节整型转8字节)。
 * 
 * @note 自动转换为本地字节序。
*/
int abcdk_mp4_read_u32to64(int fd, uint64_t *data);

/**
 * MP4读数据(8字节整型)。
 * 
 * @note 自动转换为本地字节序。
*/
int abcdk_mp4_read_u64(int fd, uint64_t *data);

/**
 * MP4读数据(N字节整型转4字节)。
 * 
 * @note 自动转换为本地字节序。
 * 
 * @param flag 标志。0 is 1byte, 1 is 2bytes,2 is 3bytes, 3 is 4bytes。
*/
int abcdk_mp4_read_nbytes_u32(int fd, int flag, uint32_t *data);

__END_DECLS

#endif //ABCDK_MP4_FILE_H
