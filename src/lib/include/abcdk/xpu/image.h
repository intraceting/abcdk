/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_IMAGE_H
#define ABCDK_XPU_IMAGE_H

#include "abcdk/xpu/types.h"


__BEGIN_DECLS

/**图像环境.*/
typedef struct _abcdk_xpu_image abcdk_xpu_image_t;

/**释放. */
void abcdk_xpu_image_free(abcdk_xpu_image_t **ctx);

/**创建. */
abcdk_xpu_image_t *abcdk_xpu_image_alloc();

/**
 * 重置.
 * 
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_image_reset(abcdk_xpu_image_t **ctx, int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align);

/**创建. */
abcdk_xpu_image_t *abcdk_xpu_image_create(int width, int height, abcdk_xpu_pixfmt_t pixfmt, int align);

/**
 * 复制.
 * 
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_image_copy(const abcdk_xpu_image_t *src, abcdk_xpu_image_t *dst);

/**宽度(像素). */
int abcdk_xpu_image_get_width(const abcdk_xpu_image_t *src);

/**高度(像素). */
int abcdk_xpu_image_get_height(const abcdk_xpu_image_t *src);

/**像素格式. */
abcdk_xpu_pixfmt_t abcdk_xpu_image_get_pixfmt(const abcdk_xpu_image_t *src);

/**
 * 上载.
 * 
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_image_upload(const uint8_t *src_data[4], const int src_linesize[4],abcdk_xpu_image_t *dst);

/**
 * 下载.
 * 
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_image_download(const abcdk_xpu_image_t *src,  uint8_t *dst_data[4], int dst_linesize[4]);

/**
 * 判断是否为空.
 * 
 * @return !0 是, 0 否.
 */
int abcdk_xpu_image_empty(const abcdk_xpu_image_t *src);


__END_DECLS

#endif // ABCDK_XPU_IMAGE_H