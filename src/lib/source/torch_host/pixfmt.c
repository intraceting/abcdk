/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/pixfmt.h"

int abcdk_torch_pixfmt_channels(int format)
{
    assert(format >= 0);

    switch (format)
    {
    case ABCDK_TORCH_PIXFMT_YUV420P:
    case ABCDK_TORCH_PIXFMT_YUV420P9:
    case ABCDK_TORCH_PIXFMT_YUV420P10:
    case ABCDK_TORCH_PIXFMT_YUV420P12:
    case ABCDK_TORCH_PIXFMT_YUV420P14:
    case ABCDK_TORCH_PIXFMT_YUV420P16:
    case ABCDK_TORCH_PIXFMT_YUV422P:
    case ABCDK_TORCH_PIXFMT_YUV422P9:
    case ABCDK_TORCH_PIXFMT_YUV422P10:
    case ABCDK_TORCH_PIXFMT_YUV422P12:
    case ABCDK_TORCH_PIXFMT_YUV422P14:
    case ABCDK_TORCH_PIXFMT_YUV422P16:
    case ABCDK_TORCH_PIXFMT_YUV444P:
    case ABCDK_TORCH_PIXFMT_YUV444P9:
    case ABCDK_TORCH_PIXFMT_YUV444P10:
    case ABCDK_TORCH_PIXFMT_YUV444P12:
    case ABCDK_TORCH_PIXFMT_YUV444P14:
    case ABCDK_TORCH_PIXFMT_YUV444P16:
    case ABCDK_TORCH_PIXFMT_NV12:
    case ABCDK_TORCH_PIXFMT_NV21:
    case ABCDK_TORCH_PIXFMT_NV16:
    case ABCDK_TORCH_PIXFMT_NV24:
    case ABCDK_TORCH_PIXFMT_NV42:
        return 3;
    case ABCDK_TORCH_PIXFMT_GRAY8:
    case ABCDK_TORCH_PIXFMT_GRAY16:
    case ABCDK_TORCH_PIXFMT_GRAYF32:
        return 1;
    case ABCDK_TORCH_PIXFMT_RGB24:
    case ABCDK_TORCH_PIXFMT_BGR24:
        return 3;
    case ABCDK_TORCH_PIXFMT_RGB32:
    case ABCDK_TORCH_PIXFMT_BGR32:
        return 4;
    default:
        return 0;
    }

    return 0;
}