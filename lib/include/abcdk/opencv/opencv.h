/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_OPENCV_OPENCV_H
#define ABCDK_OPENCV_OPENCV_H

#include "abcdk/util/ndarray.h"
#include "abcdk/util/trace.h"

#ifdef HAVE_OPENCV
#ifdef __cplusplus
#include "opencv2/opencv.hpp"
#ifdef OPENCV_ENABLE_NONFREE
#include "opencv2/xfeatures2d.hpp"
#endif // OPENCV_ENABLE_NONFREE
#endif //__cplusplus
#endif // HAVE_OPENCV

#endif //ABCDK_OPENCV_OPENCV_H