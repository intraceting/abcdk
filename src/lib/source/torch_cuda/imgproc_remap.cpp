/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/nvidia.h"

#ifdef __cuda_cuda_h__

static int _abcdk_torch_imgproc_remap_8u_cuda(int channels, int packed,
                                              uint8_t *dst, size_t dst_w, size_t dst_ws, size_t dst_h, const abcdk_torch_rect_t *dst_roi,
                                              const uint8_t *src, size_t src_w, size_t src_ws, size_t src_h, const abcdk_torch_rect_t *src_roi,
                                              const float *xmap, size_t xmap_ws, const float *ymap, size_t ymap_ws,
                                              int inter_mode)
{
    NppiSize tmp_dst_size = {0}, tmp_src_size = {0};
    NppiRect tmp_src_roi = {0};
    NppStatus npp_chk = NPP_NOT_IMPLEMENTED_ERROR;

    assert(channels == 1 || channels == 3 || channels == 4);
    assert(dst != NULL && dst_w > 0 && dst_ws > 0 && dst_h > 0);
    assert(src != NULL && src_w > 0 && src_ws > 0 && src_h > 0);
    assert(xmap != NULL && xmap_ws > 0);
    assert(ymap != NULL && ymap_ws > 0);

    tmp_dst_size.width = dst_w;
    tmp_dst_size.height = dst_h;

    tmp_src_size.width = src_w;
    tmp_src_size.height = src_h;

    tmp_src_roi.x = (src_roi ? src_roi->x : 0);
    tmp_src_roi.y = (src_roi ? src_roi->y : 0);
    tmp_src_roi.width = (src_roi ? src_roi->width : src_w);
    tmp_src_roi.height = (src_roi ? src_roi->height : src_h);

    if (channels == 1)
    {
        npp_chk = nppiRemap_8u_C1R(src, tmp_src_size, src_ws, tmp_src_roi,
                                   xmap, xmap_ws, ymap, ymap_ws,
                                   dst, dst_ws, tmp_dst_size,
                                   inter_mode);
    }
    else if (channels == 3)
    {
        npp_chk = nppiRemap_8u_C3R(src, tmp_src_size, src_ws, tmp_src_roi,
                                   xmap, xmap_ws, ymap, ymap_ws,
                                   dst, dst_ws, tmp_dst_size,
                                   inter_mode);
    }
    else if (channels == 4)
    {
        npp_chk = nppiRemap_8u_C4R(src, tmp_src_size, src_ws, tmp_src_roi,
                                   xmap, xmap_ws, ymap, ymap_ws,
                                   dst, dst_ws, tmp_dst_size,
                                   inter_mode);
    }

    if (npp_chk != NPP_SUCCESS)
        return -1;

    return 0;
}

__BEGIN_DECLS

int abcdk_torch_imgproc_remap_cuda(abcdk_torch_image_t *dst, const abcdk_torch_rect_t *dst_roi,
                                   const abcdk_torch_image_t *src, const abcdk_torch_rect_t *src_roi,
                                   const abcdk_torch_image_t *xmap, const abcdk_torch_image_t *ymap,
                                   int inter_mode)
{
    int dst_depth;

    assert(dst != NULL && src != NULL);
    // assert(dst_roi != NULL && src_roi != NULL);
    assert(xmap != NULL && ymap != NULL);
    assert(dst->pixfmt == src->pixfmt);
    assert(dst->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
           dst->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

    assert(dst->tag == ABCDK_TORCH_TAG_CUDA);
    assert(src->tag == ABCDK_TORCH_TAG_CUDA);
    assert(xmap->tag == ABCDK_TORCH_TAG_CUDA);
    assert(ymap->tag == ABCDK_TORCH_TAG_CUDA);

    dst_depth = abcdk_torch_pixfmt_channels(dst->pixfmt);

    return _abcdk_torch_imgproc_remap_8u_cuda(dst_depth, true,
                                              dst->data[0], dst->width, dst->stride[0], dst->height, dst_roi,
                                              src->data[0], src->width, src->stride[0], src->height, src_roi,
                                              (float *)xmap->data[0], xmap->stride[0], (float *)ymap->data[0], ymap->stride[0],
                                              inter_mode);
}

__END_DECLS


#endif //__cuda_cuda_h__
