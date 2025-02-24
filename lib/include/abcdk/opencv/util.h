/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_OPENCV_UTIL_H
#define ABCDK_OPENCV_UTIL_H

#include "abcdk/util/ndarray.h"
#include "abcdk/opencv/opencv.h"

__BEGIN_DECLS

/**加载图像。*/
abcdk_ndarray_t *abcdk_opencv_image_load(const char *file, int gray);


__END_DECLS

#endif //ABCDK_OPENCV_STITCHER_H