/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_OPENCV_H
#define ABCDK_TORCH_OPENCV_H

#ifdef HAVE_OPENCV
#ifdef __cplusplus
#include <opencv2/opencv.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include <opencv2/xfeatures2d.hpp>
#endif // HAVE_OPENCV_XFEATURES2D
#ifdef HAVE_OPENCV_FREETYPE
#include <opencv2/freetype.hpp>
#endif //HAVE_OPENCV_FREETYPE
#endif //__cplusplus
#endif // HAVE_OPENCV

__BEGIN_DECLS

#ifndef OPENCV_IMGPROC_HPP


#endif //OPENCV_IMGPROC_HPP


__END_DECLS

#endif //ABCDK_TORCH_OPENCV_H