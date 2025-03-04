/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/image.h"
#include "abcdk/opencv/opencv.h"

__BEGIN_DECLS

static int _abcdk_torch_image_save_to_raw(const char *dst, const abcdk_torch_image_t *src)
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

    fd = abcdk_open(dst, 1, 0, 1);
    if (fd < 0)
        return -1;

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

int abcdk_torch_image_save(const char *dst, const abcdk_torch_image_t *src)
{
    abcdk_torch_image_t *tmp_src = NULL;
    const char *dst_ext_p = NULL;
    int src_depth;
    int chk;

    assert(dst != NULL && src != NULL);
    assert(src->tag == ABCDK_TORCH_TAG_HOST);

    dst_ext_p = strrchr(dst, '.');
    
    /*如果不能通过名称识别格式，则以内存数据格式直接保存。*/
    if(dst_ext_p == NULL || abcdk_strcmp(dst_ext_p, "raw", 0) == 0)
        return _abcdk_torch_image_save_to_raw(dst,src);
    
    if(src->pixfmt != ABCDK_TORCH_PIXFMT_BGR24)
    {
        tmp_src = abcdk_torch_image_create(src->width,src->height,ABCDK_TORCH_PIXFMT_BGR24,4);
        if(!tmp_src)
            return -1;
        
        chk = abcdk_torch_image_convert(tmp_src,src);

        /*转格式成功后继续执行保存操作。*/
        if(chk == 0)
            chk = abcdk_torch_image_save(dst,tmp_src);

        abcdk_torch_image_free(&tmp_src);
        return chk;
    }

#ifndef OPENCV_IMGCODECS_HPP

    if (abcdk_strcmp(dst_ext_p, "bmp", 0) != 0)
    {
        abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具，暂时不支持使用当前(%s)格式保存。", dst_ext_p);
        return -1;
    }
 
    /*BMP图像默认是倒投影存储。这里高度传入负值，使图像正投影存储。*/
    chk = abcdk_bmp_save_file(dst, src->data[0], src->stride[0], src->width, -src->height, 24);
    if (chk != 0)
        return -1;

#else //OPENCV_IMGCODECS_HPP

    src_depth = abcdk_torch_pixfmt_channels(src->pixfmt);
    cv::Mat tmp_src2 = cv::Mat(src->height,src->width,CV_8UC(src_depth),src->data[0],src->stride[0]);

    chk = cv::imwrite(dst,tmp_src2);
    if(!chk)
        return -1;

#endif //OPENCV_IMGCODECS_HPP

    return 0;
}

__END_DECLS