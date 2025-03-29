/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_FREETYPE_H
#define ABCDK_TORCH_FREETYPE_H

#include "abcdk/util/object.h"
#include "abcdk/torch/image.h"
#include "abcdk/torch/opencv.h"

__BEGIN_DECLS

/**简单的文字引擎。*/
typedef struct _abcdk_torch_freetype abcdk_torch_freetype_t;

/**销毁。 */
void abcdk_torch_freetype_destroy(abcdk_torch_freetype_t **ctx);

/**创建。 */
abcdk_torch_freetype_t *abcdk_torch_freetype_create();

/**
 * 加载字体。
 *
 * @return 0 成功， < 0 失败。
 */
int abcdk_torch_freetype_load_font(abcdk_torch_freetype_t *ctx, const char *file, int id);

/**
 * 设置曲线的细分程度。
 *
 * @param [in] num 值。默认是 0，值越大，字体越平滑，但计算越慢。
 *
 * @return 0 成功， < 0 失败。
 */
int abcdk_torch_freetype_set_split_number(abcdk_torch_freetype_t *ctx, int num);

/**
 * 获取文字占用的宽和高(像素)。
 * 
 * @return 0 成功， < 0 失败。
 */
int abcdk_torch_freetype_get_text_size(abcdk_torch_freetype_t *ctx,
                                        abcdk_torch_size_t *size, const char *text,
                                        int height, int thickness, int *base_line);

/**
 * 向画布写文字。
 *
 * @return 0 成功， < 0 失败。
 */
int abcdk_torch_freetype_put_text_host(abcdk_torch_freetype_t *ctx,
                                       abcdk_torch_image_t *img, const char *text,
                                       abcdk_torch_point_t *org, int height, uint32_t color[4],
                                       int thickness, int line_type, uint8_t bottom_left_origin);

/**
 * 向画布写文字。
 *
 * @return 0 成功， < 0 失败。
 */
int abcdk_torch_freetype_put_text_cuda(abcdk_torch_freetype_t *ctx,
                                       abcdk_torch_image_t *img, const char *text,
                                       abcdk_torch_point_t *org, int height, uint32_t color[4],
                                       int thickness, int line_type, uint8_t bottom_left_origin);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_freetype_put_text abcdk_torch_freetype_put_text_cuda
#else // ABCDK_TORCH_USE_HOST
#define abcdk_torch_freetype_put_text abcdk_torch_freetype_put_text_host
#endif //

__END_DECLS


#endif //ABCDK_TORCH_FREETYPE_H