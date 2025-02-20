/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_IMPL_STITCHER_HXX
#define ABCDK_IMPL_STITCHER_HXX

#include "abcdk/cuda/avutil.h"
#include "abcdk/util/trace.h"

#include <string>
#include <vector>

#ifdef HAVE_OPENCV
#include "opencv2/opencv.hpp"
#ifdef OPENCV_ENABLE_NONFREE
#include "opencv2/xfeatures2d.hpp"
#endif // OPENCV_ENABLE_NONFREE
#endif // HAVE_OPENCV

#ifdef OPENCV_CORE_HPP

namespace abcdk
{

} // namespace abcdk

#endif //#ifdef OPENCV_CORE_HPP

#endif // ABCDK_IMPL_STITCHER_HXX