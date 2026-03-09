/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_IMGCODEC_H
#define ABCDK_XPU_IMGCODEC_H

#include "abcdk/xpu/image.h"


__BEGIN_DECLS

/**
 * 编码.
 * 
 * @note 默认使用JPG格式.
 * 
 * @param [in] ext 格式. 支持: .bmp, .jpg(.jpeg), .png, .tiff
*/
abcdk_object_t *abcdk_xpu_imgcodec_encode(const abcdk_xpu_image_t *src, const char *ext);

/**
 * 编码.
 * 
 * @return 0 成功, < 0 失败.
*/
int abcdk_xpu_imgcodec_encode_to_file(const abcdk_xpu_image_t *src, const char *dst, const char *ext);

/** 解码.*/
abcdk_xpu_image_t *abcdk_xpu_imgcodec_decode(const void *src, size_t size);

/** 解码.*/
abcdk_xpu_image_t *abcdk_xpu_imgcodec_decode_from_file(const char *src);

__END_DECLS

#endif // ABCDK_XPU_IMGCODEC_H