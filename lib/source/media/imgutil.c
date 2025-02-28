/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/media/imgutil.h"

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
