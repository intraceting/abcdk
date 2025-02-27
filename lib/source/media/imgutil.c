/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/imgutil.h"

static int _abcdk_meida_imgutil_fill_height(int heights[4], int height, int pixfmt)
{
    // 初始化所有平面高度为 0
    heights[0] = height; // 默认 Y 平面或单一平面高度
    heights[1] = heights[2] = heights[3] = -1;

    switch (pixfmt)
    {
    case ABCDK_MEDIA_PIXFMT_YUV410P:
        heights[1] = heights[2] = (height + 3) / 4; // UV 平面高度为 Y 的 1/4，向上取整
        break;
    case ABCDK_MEDIA_PIXFMT_YUV411P:
    case ABCDK_MEDIA_PIXFMT_YUVJ411P:
        heights[1] = heights[2] = height; // UV 平面高度与 Y 相同
        break;
    case ABCDK_MEDIA_PIXFMT_YUV420P:
    case ABCDK_MEDIA_PIXFMT_YUVJ420P:
        heights[1] = heights[2] = (height + 1) / 2; // UV 平面高度为 Y 的 1/2，向上取整
        break;
    case ABCDK_MEDIA_PIXFMT_YUV422P:
    case ABCDK_MEDIA_PIXFMT_YUVJ422P:
        heights[1] = heights[2] = height; // UV 平面高度与 Y 相同
        break;
    case ABCDK_MEDIA_PIXFMT_YUV444P:
    case ABCDK_MEDIA_PIXFMT_YUVJ444P:
        heights[1] = heights[2] = height; // UV 平面高度与 Y 相同
        break;
    case ABCDK_MEDIA_PIXFMT_YUYV422:
    case ABCDK_MEDIA_PIXFMT_UYVY422:
        // packed 格式，只有一个平面，高度已在 heights[0] 中设置
        break;
    case ABCDK_MEDIA_PIXFMT_NV12:
    case ABCDK_MEDIA_PIXFMT_NV21:
        heights[1] = (height + 1) / 2; // UV 平面交织，高度为 Y 的 1/2
        break;
    case ABCDK_MEDIA_PIXFMT_NV16:
        heights[1] = height; // UV 平面交织，高度与 Y 相同
        break;
    case ABCDK_MEDIA_PIXFMT_NV24:
    case ABCDK_MEDIA_PIXFMT_NV42:
        heights[1] = height; // UV 平面交织，高度与 Y 相同
        break;
    case ABCDK_MEDIA_PIXFMT_GRAY8:
    case ABCDK_MEDIA_PIXFMT_RGB24:
    case ABCDK_MEDIA_PIXFMT_BGR24:
    case ABCDK_MEDIA_PIXFMT_RGB32:
    case ABCDK_MEDIA_PIXFMT_BGR32:
    case ABCDK_MEDIA_PIXFMT_GRAYF32:
        // 单一平面格式，高度已在 heights[0] 中设置
        break;
    default:
        return -1; // 不支持的格式，返回错误
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

static int _abcdk_meida_imgutil_fill_stride(int stride[4], int width, int pixfmt)
{
    // 初始化所有平面步长为 0
    stride[0] = width; // 默认 Y 平面或单一平面步长
    stride[1] = stride[2] = stride[3] = -1;

    switch (pixfmt)
    {
    case ABCDK_MEDIA_PIXFMT_YUV410P:
        stride[1] = stride[2] = (width + 3) / 4; // UV 平面步长为 Y 的 1/4，向上取整
        break;
    case ABCDK_MEDIA_PIXFMT_YUV411P:
    case ABCDK_MEDIA_PIXFMT_YUVJ411P:
        stride[1] = stride[2] = (width + 3) / 4; // UV 平面步长为 Y 的 1/4，向上取整
        break;
    case ABCDK_MEDIA_PIXFMT_YUV420P:
    case ABCDK_MEDIA_PIXFMT_YUVJ420P:
        stride[1] = stride[2] = (width + 1) / 2; // UV 平面步长为 Y 的 1/2，向上取整
        break;
    case ABCDK_MEDIA_PIXFMT_YUV422P:
    case ABCDK_MEDIA_PIXFMT_YUVJ422P:
        stride[1] = stride[2] = (width + 1) / 2; // UV 平面步长为 Y 的 1/2，向上取整
        break;
    case ABCDK_MEDIA_PIXFMT_YUV444P:
    case ABCDK_MEDIA_PIXFMT_YUVJ444P:
        stride[1] = stride[2] = width; // UV 平面步长与 Y 相同
        break;
    case ABCDK_MEDIA_PIXFMT_YUYV422:
    case ABCDK_MEDIA_PIXFMT_UYVY422:
        stride[0] = width * 2; // packed YUV422，每个像素 2 字节
        break;
    case ABCDK_MEDIA_PIXFMT_NV12:
    case ABCDK_MEDIA_PIXFMT_NV21:
        stride[1] = width; // UV 平面交织，步长与 Y 相同
        break;
    case ABCDK_MEDIA_PIXFMT_NV16:
        stride[1] = width; // UV 平面交织，步长与 Y 相同
        break;
    case ABCDK_MEDIA_PIXFMT_NV24:
    case ABCDK_MEDIA_PIXFMT_NV42:
        stride[1] = width * 2; // UV 平面交织，步长为 width * 2
        break;
    case ABCDK_MEDIA_PIXFMT_GRAY8:
        stride[0] = width; // 1 字节 per pixel
        break;
    case ABCDK_MEDIA_PIXFMT_RGB24:
    case ABCDK_MEDIA_PIXFMT_BGR24:
        stride[0] = width * 3; // 3 字节 per pixel
        break;
    case ABCDK_MEDIA_PIXFMT_RGB32:
    case ABCDK_MEDIA_PIXFMT_BGR32:
        stride[0] = width * 4; // 4 字节 per pixel
        break;
    case ABCDK_MEDIA_PIXFMT_GRAYF32:
        stride[0] = width * 4; // 4 字节 per pixel (float)
        break;
    default:
        return -1; // 不支持的格式，返回错误
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

int abcdk_media_imgutil_fill_height(int heights[4], int height, int pixfmt)
{
    assert(heights != NULL && height > 0 && pixfmt >= 0);

    return _abcdk_meida_imgutil_fill_height(heights, height, pixfmt);
}

int abcdk_media_imgutil_fill_stride(int stride[4], int width, int pixfmt, int align)
{
    int block = 0;

    assert(stride != NULL && width > 0 && pixfmt >= 0);

    block = _abcdk_meida_imgutil_fill_stride(stride, width, pixfmt);
    if (block <= 0)
        return -1;

    for (int i = 0; i < block; i++)
        stride[i] = abcdk_align(stride[i], align);

    return block;
}

int abcdk_media_imgutil_fill_pointer(uint8_t *data[4], const int stride[4], int height, int pixfmt, void *buffer)
{
    int heights[4] = {0};
    int block = 0;
    int off = 0;

    assert(data != NULL && stride != NULL && height > 0 && pixfmt >= 0);

    block = abcdk_media_imgutil_fill_height(heights, height, pixfmt);
    if (block <= 0)
        return -1;

    for (int i = 0; i < block; i++)
    {
        if (stride[i] <= 0 || heights[i] <= 0)
            break;

        if (buffer)
            data[i] = ABCDK_PTR2U8PTR(buffer, off);

        off += stride[i] * heights[i];
    }

    return off;
}

int abcdk_media_imgutil_size(const int stride[4], int height, int pixfmt)
{
    uint8_t *data[4] = {0};

    assert(stride != NULL && height > 0 && pixfmt >= 0);

    return abcdk_media_imgutil_fill_pointer(data, stride, height, pixfmt, NULL);
}

int abcdk_media_imgutil_size2(int width, int height, int pixfmt, int align)
{
    int stride[4] = {0};
    int block = 0;

    block = abcdk_media_imgutil_fill_stride(stride, width, pixfmt, align);
    if (block <= 0)
        return -1;

    return abcdk_media_imgutil_size(stride, height, pixfmt);
}

void abcdk_media_imgutil_copy(uint8_t *dst_data[4], int dst_stride[4],
                              const uint8_t *src_data[4], const int src_stride[4],
                              int width, int height, int pixfmt)
{
    int real_stride[4] = {0};
    int real_height[4] = {0};

    assert(dst_data != NULL && dst_stride != NULL);
    assert(src_data != NULL && src_stride != NULL);
    assert(width > 0 && height > 0 && pixfmt >= 0);

    abcdk_media_imgutil_fill_stride(real_stride, width, pixfmt, 1);
    abcdk_media_imgutil_fill_height(real_height, height, pixfmt);

    for (int i = 0; i < 4; i++)
    {
        if (!src_data[i])
            break;

        abcdk_memcpy_2d(dst_data[i], dst_stride[i], 0, 0,
                        src_data[i], src_stride[i], 0, 0,
                        real_stride[i], real_height[i]);
    }
}
