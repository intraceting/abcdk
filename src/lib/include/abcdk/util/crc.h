/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_UTIL_CRC_H
#define ABCDK_UTIL_CRC_H

#include "abcdk/util/general.h"

__BEGIN_DECLS

/**
 * 计算CRC32值.
 * 
 * @note 默认: 未分类
*/
uint32_t abcdk_crc32(const void *data,size_t size, ...);

/**
 * 计算CRC16值.
 * 
 * @note 默认: MODBUS
*/
uint16_t abcdk_crc16(const void *data, size_t size, ...);

__END_DECLS


#endif //ABCDK_UTIL_CRC_H