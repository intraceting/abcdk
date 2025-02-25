/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/imgutil.h"

static int _abcdk_meida_image_fill_height(int heights[4], int height, int pixfmt)
{
    heights[0] = height;
    heights[1] = heights[2] = heights[3] = -1;

    switch (pixfmt)
    {
    case ABCDK_MEDIA_PIXFMT_YUV420P:
    case ABCDK_MEDIA_PIXFMT_I420:
    case ABCDK_MEDIA_PIXFMT_YV12:
    case ABCDK_MEDIA_PIXFMT_NV12:
    case ABCDK_MEDIA_PIXFMT_NV21:
        heights[1] = heights[2] = (height + 1) / 2;
        break;
    case ABCDK_MEDIA_PIXFMT_YUV422P:
    case ABCDK_MEDIA_PIXFMT_YUYV:
    case ABCDK_MEDIA_PIXFMT_UYVY:
    case ABCDK_MEDIA_PIXFMT_NV16:
    case ABCDK_MEDIA_PIXFMT_YUV411P:
        heights[1] = heights[2] = height;
        break;
    case ABCDK_MEDIA_PIXFMT_YUV444P:
    case ABCDK_MEDIA_PIXFMT_NV24:
        heights[1] = heights[2] = height;
        break;
    case ABCDK_MEDIA_PIXFMT_RGB24:
    case ABCDK_MEDIA_PIXFMT_BGR24:
    case ABCDK_MEDIA_PIXFMT_RGB32:
    case ABCDK_MEDIA_PIXFMT_BGR32:
    case ABCDK_MEDIA_PIXFMT_GRAY8:
    case ABCDK_MEDIA_PIXFMT_GRAYF32:
        heights[0] = height;
        break;
    default:
        return -1;
    }

    if (heights[3] > 0)
        return 4;
    if (heights[2] > 0)
        return 3;
    if (heights[1] > 0)
        return 2;
    if (heights[0] > 0)
        return 1;

    return 0;
}

int abcdk_media_image_fill_height(int heights[4], int height, int pixfmt)
{
    assert(heights != NULL && height > 0 && pixfmt > 0);

    return _abcdk_meida_image_fill_height(heights, height, pixfmt);
}

static int _abcdk_meida_image_fill_stride(int stride[4], int width, int pixfmt)
{
    stride[0] = width;
    stride[1] = stride[2] = stride[3] = -1;

    switch (pixfmt)
    {
    case ABCDK_MEDIA_PIXFMT_YUV420P:
    case ABCDK_MEDIA_PIXFMT_I420:
    case ABCDK_MEDIA_PIXFMT_YV12:
    case ABCDK_MEDIA_PIXFMT_NV12:
    case ABCDK_MEDIA_PIXFMT_NV21:
        stride[1] = stride[2] = (width + 1) / 2;
        break;
    case ABCDK_MEDIA_PIXFMT_YUV422P:
    case ABCDK_MEDIA_PIXFMT_YUYV:
    case ABCDK_MEDIA_PIXFMT_UYVY:
    case ABCDK_MEDIA_PIXFMT_NV16:
        stride[1] = stride[2] = (width + 1) / 2;
        break;
    case ABCDK_MEDIA_PIXFMT_YUV444P:
    case ABCDK_MEDIA_PIXFMT_NV24:
        stride[1] = stride[2] = width;
        break;
    case ABCDK_MEDIA_PIXFMT_YUV411P:
        stride[1] = stride[2] = (width + 3) / 4;
        break;
    case ABCDK_MEDIA_PIXFMT_RGB24:
    case ABCDK_MEDIA_PIXFMT_BGR24:
        stride[0] = width * 3;
        break;
    case ABCDK_MEDIA_PIXFMT_RGB32:
    case ABCDK_MEDIA_PIXFMT_BGR32:
        stride[0] = width * 4;
        break;
    case ABCDK_MEDIA_PIXFMT_GRAY8:
        stride[0] = width;
        break;
    case ABCDK_MEDIA_PIXFMT_GRAYF32:
        stride[0] = width * 4;
        break;
    default:
        return -1;
    }

    if (stride[3] > 0)
        return 4;
    if (stride[2] > 0)
        return 3;
    if (stride[1] > 0)
        return 2;
    if (stride[0] > 0)
        return 1;

    return 0;
}

int abcdk_media_image_fill_stride(int stride[4], int width, int pixfmt, int align)
{
    int block = 0;

    assert(stride != NULL && width > 0 && pixfmt > 0);

    block = _abcdk_meida_image_fill_stride(stride, width, pixfmt);
    if (block <= 0)
        return -1;

    for (int i = 0; i < block; i++)
        stride[i] = abcdk_align(stride[i], align);

    return block;
}

int abcdk_media_image_fill_pointer(uint8_t *data[4], const int stride[4], int height, int pixfmt, void *buffer)
{
    int heights[4] = {0};
    int block = 0;
    int off = 0;

    assert(data != NULL && stride != NULL && height > 0 && pixfmt > 0);

    block = abcdk_media_image_fill_height(heights, height, pixfmt);
    if (block <= 0)
        return -1;

    for (int i = 0; i < block; i++)
    {
        if (stride[i] <= 0 || heights[i] <= 0)
            break;

        if (buffer)
            data[i] = ABCDK_PTR2I8PTR(buffer, off);

        off += stride[i] * heights[i];
    }

    return off;
}

int abcdk_media_image_size(const int stride[4], int height, int pixfmt)
{
    uint8_t *data[4] = {0};

    assert(stride != NULL && height > 0 && pixfmt > 0);

    return abcdk_media_image_fill_pointer(data, stride, height, pixfmt, NULL);
}

int abcdk_media_image_size2(int width, int height, int pixfmt, int align)
{
    int stride[4] = {0};
    int block = 0;

    block = abcdk_media_image_fill_stride(stride, width, pixfmt, align);
    if (block <= 0)
        return -1;

    return abcdk_media_image_size(stride, height, pixfmt);
}

void abcdk_media_image_copy(uint8_t *dst_data[4], int dst_stride[4],
                            const uint8_t *src_data[4], const int src_stride[4],
                            int width, int height, int pixfmt)
{
    int real_stride[4] = {0};
    int real_height[4] = {0};
    int chk;

    assert(dst_data != NULL && dst_stride != NULL);
    assert(src_data != NULL && src_stride != NULL);
    assert(width > 0 && height > 0 && pixfmt > 0);

    abcdk_media_image_fill_stride(real_stride, width, pixfmt, 1);
    abcdk_media_image_fill_height(real_height, height, pixfmt);

    for (int i = 0; i < 4; i++)
    {
        if (!src_data[i])
            break;

        abcdk_memcpy_2d(dst_data[i], dst_stride[i], 0, 0,
                        src_data[i], src_stride[i], 0, 0,
                        real_stride[i], real_height[i]);
    }
}