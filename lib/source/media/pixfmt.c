/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/pixfmt.h"

int abcdk_media_pixfmt_channels(int format)
{
    assert(format >= 0);

    switch (format)
    {
    case ABCDK_MEDIA_PIXFMT_YUV410P:
    case ABCDK_MEDIA_PIXFMT_YUV411P:
    case ABCDK_MEDIA_PIXFMT_YUV420P:
    case ABCDK_MEDIA_PIXFMT_YUV422P:
    case ABCDK_MEDIA_PIXFMT_YUV444P:
    case ABCDK_MEDIA_PIXFMT_YUVJ411P:
    case ABCDK_MEDIA_PIXFMT_YUVJ420P:
    case ABCDK_MEDIA_PIXFMT_YUVJ422P:
    case ABCDK_MEDIA_PIXFMT_YUVJ444P:
    case ABCDK_MEDIA_PIXFMT_YUYV422:
    case ABCDK_MEDIA_PIXFMT_UYVY422:
    case ABCDK_MEDIA_PIXFMT_NV12:
    case ABCDK_MEDIA_PIXFMT_NV21:
    case ABCDK_MEDIA_PIXFMT_NV16:
    case ABCDK_MEDIA_PIXFMT_NV24:
    case ABCDK_MEDIA_PIXFMT_NV42:
        return 3; // Y, U, V 三个通道
    case ABCDK_MEDIA_PIXFMT_GRAY8:
    case ABCDK_MEDIA_PIXFMT_GRAYF32:
        return 1; // 灰度只有一个通道
    case ABCDK_MEDIA_PIXFMT_RGB24:
    case ABCDK_MEDIA_PIXFMT_BGR24:
        return 3; // R, G, B 三个通道
    case ABCDK_MEDIA_PIXFMT_RGB32:
    case ABCDK_MEDIA_PIXFMT_BGR32:
        return 4; // R, G, B, A 四个通道
    default:
        return -1; // 不支持的格式
    }

    return -1;
}

int abcdk_media_pixfmt_bitwidth(int format, int plane)
{
    assert(format >= 0);

    switch (format)
    {
    case ABCDK_MEDIA_PIXFMT_YUV410P:
    case ABCDK_MEDIA_PIXFMT_YUV411P:
    case ABCDK_MEDIA_PIXFMT_YUV420P:
    case ABCDK_MEDIA_PIXFMT_YUV422P:
    case ABCDK_MEDIA_PIXFMT_YUV444P:
    case ABCDK_MEDIA_PIXFMT_YUVJ411P:
    case ABCDK_MEDIA_PIXFMT_YUVJ420P:
    case ABCDK_MEDIA_PIXFMT_YUVJ422P:
    case ABCDK_MEDIA_PIXFMT_YUVJ444P:
        return plane == 0 ? 8 : 8; // Y 8-bit, UV 8-bit (subsampled)

    case ABCDK_MEDIA_PIXFMT_YUYV422:
    case ABCDK_MEDIA_PIXFMT_UYVY422:
        return 16; // Packed YUV422 (YUV per 2 pixels)

    case ABCDK_MEDIA_PIXFMT_NV12:
    case ABCDK_MEDIA_PIXFMT_NV21:
    case ABCDK_MEDIA_PIXFMT_NV16:
    case ABCDK_MEDIA_PIXFMT_NV24:
    case ABCDK_MEDIA_PIXFMT_NV42:
        return plane == 0 ? 8 : 8; // NV12/NV21/NV16/NV24 8-bit per channel

    case ABCDK_MEDIA_PIXFMT_GRAY8:
        return 8; // Grayscale 8-bit

    case ABCDK_MEDIA_PIXFMT_RGB24:
    case ABCDK_MEDIA_PIXFMT_BGR24:
        return 8; // 8-bit per channel (3 channels)

    case ABCDK_MEDIA_PIXFMT_RGB32:
    case ABCDK_MEDIA_PIXFMT_BGR32:
        return 8; // 8-bit per channel (4 channels)

    case ABCDK_MEDIA_PIXFMT_GRAYF32:
        return 32; // 32-bit floating point grayscale

    default:
        return 0; // Unsupported format
    }

    return 0;
}
