/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_OPENCV_UTIL_H
#define ABCDK_OPENCV_UTIL_H

#include "abcdk/media/image.h"
#include "abcdk/opencv/opencv.h"

__BEGIN_DECLS

/**加载图像。*/
abcdk_media_image_t *abcdk_opencv_image_load(const char *src, int gray);

/**
 * 保存图像。
 * 
 * @return 0 成功， < 0 失败。
*/
int abcdk_opencv_image_save(const char *dst, abcdk_media_image_t *src);


__END_DECLS

#endif //ABCDK_OPENCV_STITCHER_H