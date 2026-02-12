/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
*/
#ifndef ABCDK_QRCODE_UTIL_H
#define ABCDK_QRCODE_UTIL_H

#include "abcdk/util/trace.h"
#include "abcdk/util/bmp.h"
#include "abcdk/util/object.h"
#include "abcdk/qrcode/qrcode.h"

__BEGIN_DECLS

/**
 * 编码.
 * 
 * @note 编码使用Y8格式.
 * 
*/
abcdk_object_t *abcdk_qrcode_encode(const char *data, size_t size, int level, int scale, int margin);

/**
 * 编码并保存.
 * 
 * @note 保存为BMP格式.
 * 
 * @return 0 成功, -1 失败.
*/
int abcdk_qrcode_encode_save(const char *dst, const char *data, size_t size, int level, int scale, int margin);

__END_DECLS

#endif //ABCDK_QRCODE_UTIL_H