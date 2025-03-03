/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_TORCH_H
#define ABCDK_TORCH_TORCH_H

#include "abcdk/util/general.h"
#include "abcdk/util/object.h"
#include "abcdk/util/trace.h"
#include "abcdk/util/bmp.h"
#include "abcdk/opencv/opencv.h"
#include "abcdk/ffmpeg/ffmpeg.h"

/**主机对象。*/
#define ABCDK_TORCH_TAG_HOST ABCDK_FOURCC_MKTAG('h', 'o', 's', 't')

/**CUDA对象。*/
#define ABCDK_TORCH_TAG_CUDA ABCDK_FOURCC_MKTAG('C', 'U', 'D', 'A')

/**点。*/
typedef struct _abcdk_torch_point
{
    int x;
    int y;
} abcdk_torch_point_t;

/**尺寸。*/
typedef struct _abcdk_torch_size
{
    int width;
    int height;
} abcdk_torch_size_t;

/**区域。*/
typedef struct _abcdk_torch_rect
{
    int x;
    int y;
    int width;
    int height;
} abcdk_torch_rect_t;

#endif //ABCDK_TORCH_TORCH_H