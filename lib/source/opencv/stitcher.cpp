/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/opencv/stitcher.h"
#include "stitcher_general.hxx"

#ifdef OPENCV_CORE_HPP

/**简单的全景拼接引擎。*/
struct _abcdk_stitcher
{
    /**/
    abcdk::opencv::stitcher_general *impl_ctx;

}; // abcdk_stitcher_t;

void abcdk_stitcher_destroy(abcdk_stitcher_t **ctx)
{
    abcdk_stitcher_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    delete ctx_p->impl_ctx;

    abcdk_heap_free(ctx_p);
}

abcdk_stitcher_t *abcdk_stitcher_create()
{
    abcdk_stitcher_t *ctx;

    ctx = (abcdk_stitcher_t *)abcdk_heap_alloc(sizeof(abcdk_stitcher_t));
    if (!ctx)
        return NULL;

    ctx->impl_ctx = new abcdk::opencv::stitcher_general();
    if (!ctx->impl_ctx)
        goto ERR;

    return ctx;

ERR:

    abcdk_stitcher_destroy(&ctx);
    return NULL;
}

abcdk_object_t *abcdk_stitcher_metadata_dump(abcdk_stitcher_t *ctx, const char *magic)
{
    abcdk_object_t *out = NULL;
    std::string out_data;
    int chk;

    assert(ctx != NULL && magic != NULL);

    chk = abcdk::opencv::stitcher_general::Dump(out_data, *ctx->impl_ctx, magic);
    if (chk != 0)
        return NULL;

    out = abcdk_object_copyfrom(out_data.c_str(), out_data.length());
    if (!out)
        return NULL;

    return out;
}

int abcdk_stitcher_metadata_load(abcdk_stitcher_t *ctx, const char *magic, const char *data)
{
    int chk;

    assert(ctx != NULL && magic != NULL && data != NULL);

    chk = abcdk::opencv::stitcher_general::Load(data, *ctx->impl_ctx, magic);
    if (chk == 0)
        return 0;
    else if (chk == -127)
        return -127;

    return -1;
}

int abcdk_stitcher_estimate_transform(abcdk_stitcher_t *ctx, int count, abcdk_media_frame_t *img[], abcdk_media_frame_t *mask[], float good_threshold)
{
    std::vector<cv::Mat> tmp_imgs;
    std::vector<cv::Mat> tmp_masks;
    int chk;

    assert(ctx != NULL && count >= 2 && img != NULL && mask != NULL);

    // tmp_imgs.resize(count);
    // tmp_masks.resize(count);

    // for (int i = 0; i < count; i++)
    // {
    //     auto &dst_img = tmp_imgs[i];
    //     auto &dst_mask = tmp_masks[i];
    //     auto &src_img = img[i];
    //     auto &src_mask = mask[i];

    //     assert(src_img != NULL);
    //     assert(src_img->tag == ABCDK_MEDIA_FRAME_TAG_HOST);

    //     int src_img_depth = abcdk_media_pixfmt_channels(src_img->pixfmt);

    //     dst_img.create(src_img->height, src_img->width, CV_8UC(src_img_depth));
    //     if (dst_img.empty())
    //         return -1;

    //     abcdk_memcpy_2d(dst_img.data, dst_img.step, 0, 0,
    //                     src_img->data[0], src_img->stride[0], 0, 0,
    //                     src_img_depth * src_img->width, src_img->height);

    //     if (src_mask)
    //     {
    //         assert(src_mask != NULL);
    //         assert(src_mask->tag == ABCDK_MEDIA_FRAME_TAG_HOST);

    //         int src_mask_depth = abcdk_media_pixfmt_channels(src_mask->pixfmt);

    //         dst_mask.create(src_mask->height, src_mask->width, CV_8UC(src_mask_depth));
    //         if (dst_mask.empty())
    //             return -1;

    //         abcdk_memcpy_2d(dst_mask.data, dst_mask.step, 0, 0,
    //                         src_mask->data[0], src_mask->stride[0], 0, 0,
    //                         src_mask_depth * src_mask->width, src_mask->height);
    //     }
    // }

    chk = ctx->impl_ctx->EstimateTransform(tmp_imgs, tmp_masks, good_threshold);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_stitcher_build_panorama_param(abcdk_stitcher_t *ctx)
{
    assert(ctx != NULL);

    ctx->impl_ctx->BuildPanoramaParam();

    return 0;
}

int abcdk_stitcher_compose_panorama(abcdk_stitcher_t *ctx, abcdk_media_frame_t **out, int count, abcdk_media_frame_t *img[])
{
    std::vector<cv::Mat> tmp_imgs;
    cv::Mat tmp_out;
    int chk;

    assert(ctx != NULL && out != NULL && count >= 2 && img != NULL);

    // tmp_imgs.resize(count);

    // for (int i = 0; i < count; i++)
    // {
    //     auto &dst_img = tmp_imgs[i];
    //     auto &src_img = img[i];

    //     assert(src_img != NULL);
    //     assert(src_img->tag == ABCDK_MEDIA_FRAME_TAG_HOST);

    //     int src_img_depth = abcdk_media_pixfmt_channels(src_img->pixfmt);

    //     dst_img.create(src_img->height, src_img->width, CV_8UC(src_img_depth));
    //     if (dst_img.empty())
    //         return -1;

    //     abcdk_memcpy_2d(dst_img.data, dst_img.step, 0, 0,
    //                     src_img->data[0], src_img->stride[0], 0, 0,
    //                     src_img_depth * src_img->width, src_img->height);
    // }

    // chk = ctx->impl_ctx->ComposePanorama(tmp_out,tmp_imgs);
    // if(chk != 0)
    //     return -1;

    // uint8_t *out_data[4] = {NULL, NULL, NULL, NULL};
    // int out_stride[4] = {-1, -1, -1, -1};

    // out_data[0] = tmp_out.data;
    // out_stride[0] = tmp_out.step;

    // *out = abcdk_media_frame_clone2((const uint8_t **)out_data,out_stride,tmp_out.cols,tmp_out.rows,ABCDK_MEDIA_PIXFMT_BGR24);

    return 0;
}

#else // OPENCV_CORE_HPP

void abcdk_stitcher_destroy(abcdk_stitcher_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
}

abcdk_stitcher_t *abcdk_stitcher_create()
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return NULL;
}

abcdk_object_t *abcdk_stitcher_metadata_dump(abcdk_stitcher_t *ctx, const char *magic)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return NULL;
}

int abcdk_stitcher_metadata_load(abcdk_stitcher_t *ctx, const char *magic, const char *data)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return -1;
}

int abcdk_stitcher_estimate_transform(abcdk_stitcher_t *ctx, int count, abcdk_media_frame_t *img[], abcdk_media_frame_t *mask[], float good_threshold)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return -1;
}

int abcdk_stitcher_build_panorama_param(abcdk_stitcher_t *ctx)
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return -1;
}

int abcdk_stitcher_compose_panorama(abcdk_stitcher_t *ctx, abcdk_media_frame_t **out, int count, abcdk_media_frame_t *img[])
{
    abcdk_trace_printf(LOG_WARNING, "当前环境在构建时未包含OpenCV工具。");
    return -1;
}

#endif // OPENCV_CORE_HPP
