/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/freetype.h"
#include "abcdk/torch/nvidia.h"

__BEGIN_DECLS

#ifdef _OPENCV_FREETYPE_H_

int abcdk_torch_freetype_put_text_cuda(abcdk_torch_freetype_t *ctx,
                                       abcdk_torch_image_t *img, const char *text,
                                       abcdk_torch_point_t *org, int height, uint32_t color[4],
                                       int thickness, int line_type, uint8_t bottom_left_origin)
{
    abcdk_torch_image_t *tmp_img;
    int chk;

    assert(ctx != NULL && img != NULL && text != NULL && org != NULL && height >= 0 && line_type >= 0 && color != NULL);
    assert(img->tag == ABCDK_TORCH_TAG_CUDA);

    tmp_img = abcdk_torch_image_clone_cuda(1,img);
    if(!tmp_img)
        return -2;

    chk = abcdk_torch_freetype_put_text_host(ctx,tmp_img,text,org,height,color,thickness,line_type,bottom_left_origin);
    if(chk != 0)
        return -1;

    abcdk_torch_image_copy_cuda(img,tmp_img);
    abcdk_torch_image_free_cuda(&tmp_img);

    return 0;
}

#else //_OPENCV_FREETYPE_H_

int abcdk_torch_freetype_put_text_cuda(abcdk_torch_freetype_t *ctx,
                                       abcdk_torch_image_t *img, const char *text,
                                       abcdk_torch_point_t *org, int height, uint32_t color[4],
                                       int thickness, int line_type, uint8_t bottom_left_origin)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

#endif // _OPENCV_FREETYPE_H_

__END_DECLS
