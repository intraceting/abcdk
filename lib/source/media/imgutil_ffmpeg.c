/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/imgutil.h"

#ifdef AVUTIL_AVUTIL_H

int abcdk_media_imgutil_fill_height(int heights[4], int height, int pixfmt)
{
    assert(heights != NULL && height > 0 && pixfmt >= 0);

    return abcdk_avimage_fill_height(heights, height, abcdk_media_pixfmt_to_ffmpeg(pixfmt));
}

int abcdk_media_imgutil_fill_stride(int stride[4], int width, int pixfmt, int align)
{
    assert(stride != NULL && width > 0 && pixfmt >= 0);

    return abcdk_avimage_fill_stride(stride,width, abcdk_media_pixfmt_to_ffmpeg(pixfmt),align);
}

int abcdk_media_imgutil_fill_pointer(uint8_t *data[4], const int stride[4], int height, int pixfmt, void *buffer)
{
    assert(data != NULL && stride != NULL && height > 0 && pixfmt >= 0);

    return abcdk_avimage_fill_pointer(data,stride,height,abcdk_media_pixfmt_to_ffmpeg(pixfmt),buffer);
}

int abcdk_media_imgutil_size(const int stride[4], int height, int pixfmt)
{
    assert(stride != NULL && height > 0 && pixfmt >= 0);

    return abcdk_avimage_size(stride,height,abcdk_media_pixfmt_to_ffmpeg(pixfmt));
}

int abcdk_media_imgutil_size2(int width, int height, int pixfmt, int align)
{
    assert(width > 0 && height > 0 && pixfmt >= 0);

    return abcdk_avimage_size2(width, height, abcdk_media_pixfmt_to_ffmpeg(pixfmt), align);
}

#else // AVUTIL_AVUTIL_H

int abcdk_media_imgutil_fill_height(int heights[4], int height, int pixfmt)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含FFmpeg工具。");
    return -1;
}

int abcdk_media_imgutil_fill_stride(int stride[4], int width, int pixfmt, int align)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含FFmpeg工具。");
    return -1;
}

int abcdk_media_imgutil_fill_pointer(uint8_t *data[4], const int stride[4], int height, int pixfmt, void *buffer)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含FFmpeg工具。");
    return -1;
}

int abcdk_media_imgutil_size(const int stride[4], int height, int pixfmt)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含FFmpeg工具。");
    return -1;
}

int abcdk_media_imgutil_size2(int width, int height, int pixfmt, int align)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含FFmpeg工具。");
    return -1;
}

#endif // AVUTIL_AVUTIL_H