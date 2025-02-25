/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/pixfmt.h"

int abcdk_media_pixfmt_channels(int pixfmt)
{
    assert(pixfmt > 0);

    switch (pixfmt)
    {
    case ABCDK_MEDIA_PIXFMT_YUV420P:
    case ABCDK_MEDIA_PIXFMT_I420:
    case ABCDK_MEDIA_PIXFMT_YV12:
    case ABCDK_MEDIA_PIXFMT_NV12:
    case ABCDK_MEDIA_PIXFMT_NV21:
    case ABCDK_MEDIA_PIXFMT_YUV422P:
    case ABCDK_MEDIA_PIXFMT_YUYV:
    case ABCDK_MEDIA_PIXFMT_UYVY:
    case ABCDK_MEDIA_PIXFMT_NV16:
    case ABCDK_MEDIA_PIXFMT_YUV444P:
    case ABCDK_MEDIA_PIXFMT_NV24:
    case ABCDK_MEDIA_PIXFMT_YUV411P:
        return 3;
    case ABCDK_MEDIA_PIXFMT_RGB24:
    case ABCDK_MEDIA_PIXFMT_BGR24:
        return 3;
    case ABCDK_MEDIA_PIXFMT_RGB32:
    case ABCDK_MEDIA_PIXFMT_BGR32:
        return 4;
    case ABCDK_MEDIA_PIXFMT_GRAY8:
        return 1;
    case ABCDK_MEDIA_PIXFMT_GRAYF32:
        return 4;
    default:
        return -1;
    }

    return 0;
}
