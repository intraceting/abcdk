/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/image.h"

int abcdk_torch_image_dump_host(const char *dst, const abcdk_torch_image_t *src)
{
    int src_size = 0;
    int src_height[4];
    int src_plane = 0;
    int fd = -1;
    int chk_size = 0;
    int chk;

    src_size = abcdk_torch_imgutil_size(src->stride, src->height, src->pixfmt);
    src_plane = abcdk_torch_imgutil_fill_height(src_height, src->height, src->pixfmt);

    if (src_plane <= 0)
        return -1;

    /*创建需要的路径。*/
    abcdk_mkdir(dst,0755);

    fd = abcdk_open(dst, 1, 0, 1);
    if (fd < 0)
        return -1;

    chk = ftruncate(fd, 0);
    if (chk != 0)
    {
        abcdk_closep(&fd);
        return -1;
    }

    for (int i = 0; i < src_plane; i++)
    {
        chk = abcdk_write(fd, src->data[i], src->stride[i] * src_height[i]);

        if (chk > 0)
            chk_size += chk;

        if (chk != src->stride[i] * src_height[i])
            break;
    }

    abcdk_closep(&fd);

    return (chk_size == src_size ? 0 : -1);
}