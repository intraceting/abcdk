/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgutil.h"

static int _abcdk_torch_imgutil_fill_size(int stride[4], int heights[4], int width, int height, int pixfmt)
{
    stride[0] = stride[1] = stride[2] = stride[3] = 0;
    heights[0] = heights[1] = heights[2] = heights[3] = 0;

    switch (pixfmt)
    {
    case ABCDK_TORCH_PIXFMT_YUV420P:
    {
        width = abcdk_align(width, 2);//宽度较正为偶数。
        height = abcdk_align(height, 2);//高度较正为偶数。

        stride[0] = width;
        stride[1] = width / 2;
        stride[2] = width / 2;
        heights[0] = height;
        heights[1] = height / 2;
        heights[2] = height / 2;
    }
    break;
    case ABCDK_TORCH_PIXFMT_YUV420P9:
    case ABCDK_TORCH_PIXFMT_YUV420P10:
    case ABCDK_TORCH_PIXFMT_YUV420P12:
    case ABCDK_TORCH_PIXFMT_YUV420P14:
    case ABCDK_TORCH_PIXFMT_YUV420P16:
    {
        width = abcdk_align(width, 2);//宽度较正为偶数。
        height = abcdk_align(height, 2);//高度较正为偶数。

        stride[0] = width * 2; // 2 bytes.
        stride[1] = width / 2 * 2; // 2 bytes.
        stride[2] = width / 2 * 2; // 2 bytes.
        heights[0] = height;
        heights[1] = height / 2;
        heights[2] = height / 2;
    }
    break;
    case ABCDK_TORCH_PIXFMT_YUV422P:
    {
        width = abcdk_align(width, 2);//宽度较正为偶数。

        stride[0] = width;
        stride[1] = width / 2;
        stride[2] = width / 2;
        heights[0] = height;
        heights[1] = height;
        heights[2] = height;
    }
    break;
    case ABCDK_TORCH_PIXFMT_YUV422P9:
    case ABCDK_TORCH_PIXFMT_YUV422P10:
    case ABCDK_TORCH_PIXFMT_YUV422P12:
    case ABCDK_TORCH_PIXFMT_YUV422P14:
    case ABCDK_TORCH_PIXFMT_YUV422P16:
    {
        width = abcdk_align(width, 2);//宽度较正为偶数。

        stride[0] = width * 2; // 2 bytes.
        stride[1] = width / 2 * 2; // 2 bytes.
        stride[2] = width / 2 * 2; // 2 bytes.
        heights[0] = height;
        heights[1] = height;
        heights[2] = height;
    }
    break;
    case ABCDK_TORCH_PIXFMT_YUV444P:
    {
        stride[0] = width;
        stride[1] = width;
        stride[2] = width;
        heights[0] = height;
        heights[1] = height;
        heights[2] = height;
    }
    break;
    case ABCDK_TORCH_PIXFMT_YUV444P9:
    case ABCDK_TORCH_PIXFMT_YUV444P10:
    case ABCDK_TORCH_PIXFMT_YUV444P12:
    case ABCDK_TORCH_PIXFMT_YUV444P14:
    case ABCDK_TORCH_PIXFMT_YUV444P16:
    {
        stride[0] = width * 2; // 2 bytes.
        stride[1] = width * 2; // 2 bytes.
        stride[2] = width * 2; // 2 bytes.
        heights[0] = height;
        heights[1] = height;
        heights[2] = height;
    }
    break;
    case ABCDK_TORCH_PIXFMT_NV12: // YUV 4:2:0
    case ABCDK_TORCH_PIXFMT_NV21: // YUV 4:2:0
    {
        width = abcdk_align(width, 2);//宽度较正为偶数。
        height = abcdk_align(height, 2);//高度较正为偶数。

        stride[0] = width;
        stride[1] = width; // U+V | V+U.
        heights[0] = height;
        heights[1] = height / 2;
    }
    break;
    case ABCDK_TORCH_PIXFMT_P016: // YUV 4:2:0
    {
        width = abcdk_align(width, 2);//宽度较正为偶数。
        height = abcdk_align(height, 2);//高度较正为偶数。

        stride[0] = width * 2; // 2 bytes.
        stride[1] = width * 2; // 2 bytes, U+V | V+U. 
        heights[0] = height;
        heights[1] = height / 2;
    }
    break;
    case ABCDK_TORCH_PIXFMT_NV16: // YUV 4:2:2
    {
        width = abcdk_align(width, 2);//宽度较正为偶数。
        height = abcdk_align(height, 2);//高度较正为偶数。

        stride[0] = width;
        stride[1] = width ; // U+V.
        heights[0] = height;
        heights[1] = height;
    }
    break;
    case ABCDK_TORCH_PIXFMT_NV24: //YUV 4:4:4 
    case ABCDK_TORCH_PIXFMT_NV42: //YUV 4:4:4 
    {
        width = abcdk_align(width, 2);//宽度较正为偶数。
        height = abcdk_align(height, 2);//高度较正为偶数。

        stride[0] = width;
        stride[1] = width; //  U+V | V+U.
        heights[0] = height;
        heights[1] = height;
    }
    break;
    case ABCDK_TORCH_PIXFMT_GRAY8:
    {
        stride[0] = width;
        heights[0] = height;
    }
    break;
    case ABCDK_TORCH_PIXFMT_GRAY16:
    {
        stride[0] = width * 2; // 2 bytes.
        heights[0] = height;
    }
    break;
    case ABCDK_TORCH_PIXFMT_GRAYF32:
    {
        stride[0] = width * 4; // 4 bytes.
        heights[0] = height;
    }
    break;
    case ABCDK_TORCH_PIXFMT_RGB24:
    case ABCDK_TORCH_PIXFMT_BGR24:
    {
        stride[0] = width * 3; // 1 bytes.
        heights[0] = height;
    }
    break;
    case ABCDK_TORCH_PIXFMT_RGB32:
    case ABCDK_TORCH_PIXFMT_BGR32:
    {
        stride[0] = width * 4; // 1 bytes.
        heights[0] = height;
    }
    break;
    default:
        return 0;
    }

    for (int i = 0; i < 4; i++)
    {
        if (width > 0 && height > 0)
        {
            if (stride[4 - i - 1] > 0 && heights[4 - i - 1] > 0)
                return 4 - i;
        }
        else if (width > 0)
        {
            if (stride[4 - i - 1] > 0)
                return 4 - i;
        }
        else if (height > 0)
        {
            if (heights[4 - i - 1] > 0)
                return 4 - i;
        }
    }

    return 0;
}

int abcdk_torch_imgutil_fill_height(int heights[4], int height, int pixfmt)
{
    int stride[4];

    assert(heights != NULL && height > 0 && pixfmt >= 0);

#ifdef AVUTIL_AVUTIL_H
    return abcdk_avimage_fill_height(heights, height, abcdk_torch_pixfmt_convert_to_ffmpeg(pixfmt));
#else //AVUTIL_AVUTIL_H
    return _abcdk_torch_imgutil_fill_size(stride, heights, 0, height, pixfmt);
#endif //AVUTIL_AVUTIL_H

}

int abcdk_torch_imgutil_fill_stride(int stride[4], int width, int pixfmt, int align)
{
    int heights[4];
    int chk;

    assert(stride != NULL && width > 0 && pixfmt >= 0);

#ifdef AVUTIL_AVUTIL_H
    return abcdk_avimage_fill_stride(stride, width, abcdk_torch_pixfmt_convert_to_ffmpeg(pixfmt), align);
#else //AVUTIL_AVUTIL_H
    chk = _abcdk_torch_imgutil_fill_size(stride, heights, width, 0, pixfmt);
    if (chk <= 0)
        return 0;

    for (int i = 0; i < chk; i++)
    {
        stride[i] = abcdk_align(stride[i], align);
    }

    return chk;
#endif //AVUTIL_AVUTIL_H
}

int abcdk_torch_imgutil_fill_pointer(uint8_t *data[4], const int stride[4], int height, int pixfmt, void *buffer)
{
    int heights[4];
    int off = 0;
    int chk;

    assert(data != NULL && stride != NULL && height > 0 && pixfmt >= 0);

#ifdef AVUTIL_AVUTIL_H
    return abcdk_avimage_fill_pointer(data,stride,height,abcdk_torch_pixfmt_convert_to_ffmpeg(pixfmt),buffer);
#else //AVUTIL_AVUTIL_H
    chk = abcdk_torch_imgutil_fill_height(heights, height, pixfmt);
    if (chk <= 0)
        return -1;

    for (int i = 0; i < chk; i++)
    {
        data[i] = (buffer ? ABCDK_PTR2U8PTR(buffer, off) : NULL);
        off += stride[i] * heights[i];
    }

    return off;
#endif //AVUTIL_AVUTIL_H
}

int abcdk_torch_imgutil_size(const int stride[4], int height, int pixfmt)
{
    uint8_t *data[4];

    assert(stride != NULL && height > 0 && pixfmt >= 0);

    return abcdk_torch_imgutil_fill_pointer(data,stride,height,pixfmt,NULL);
}

int abcdk_torch_imgutil_size2(int width, int height, int pixfmt, int align)
{
    int stride[4];
    int chk;

    assert(width > 0 && height > 0 && pixfmt >= 0);

    chk = abcdk_torch_imgutil_fill_stride(stride, width, pixfmt, align);
    if(chk <= 0)
        return -1;

    return abcdk_torch_imgutil_size(stride, height, pixfmt);
}

void abcdk_torch_imgutil_copy(uint8_t *dst_data[4], int dst_stride[4],
                              const uint8_t *src_data[4], const int src_stride[4],
                              int width, int height, int pixfmt)
{
    int real_stride[4] = {0};
    int real_height[4] = {0};
    int chk, chk_plane,chk_stride, chk_height;

    assert(dst_data != NULL && dst_stride != NULL);
    assert(src_data != NULL && src_stride != NULL);
    assert(width > 0 && height > 0 && pixfmt >= 0);

    chk_stride = abcdk_torch_imgutil_fill_stride(real_stride, width, pixfmt, 1);
    chk_height = abcdk_torch_imgutil_fill_height(real_height, height, pixfmt);
    chk_plane = ABCDK_MIN(chk_stride, chk_height);

    for (int i = 0; i < chk_plane; i++)
    {
        abcdk_memcpy_2d(dst_data[i], dst_stride[i], 0, 0,
                        src_data[i], src_stride[i], 0, 0,
                        real_stride[i], real_height[i]);
    }
}
