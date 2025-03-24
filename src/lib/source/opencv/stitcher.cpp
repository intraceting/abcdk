/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/opencv/stitcher.h"
#include "stitcher_cpu.hxx"
#include "stitcher_cuda.hxx"


__BEGIN_DECLS


#ifdef OPENCV_STITCHING_STITCHER_HPP

/**简单的全景拼接引擎。*/
struct _abcdk_opencv_stitcher
{
    /**标签。*/
    uint32_t tag;

    /**引擎。*/
    abcdk::opencv::stitcher *impl_ctx;

}; // abcdk_opencv_stitcher_t;

void abcdk_opencv_stitcher_destroy(abcdk_opencv_stitcher_t **ctx)
{
    abcdk_opencv_stitcher_t *ctx_p;

    if (!ctx || !*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    if(ctx_p->tag == ABCDK_TORCH_TAG_HOST)
        delete (abcdk::opencv::stitcher_cpu*)ctx_p->impl_ctx;
    else if(ctx_p->tag == ABCDK_TORCH_TAG_CUDA)
        delete (abcdk::opencv::stitcher_cuda*)ctx_p->impl_ctx;
    
    abcdk_heap_free(ctx_p);
}

abcdk_opencv_stitcher_t *abcdk_opencv_stitcher_create(uint32_t tag)
{
    abcdk_opencv_stitcher_t *ctx;
    
    assert(tag == ABCDK_TORCH_TAG_HOST || tag == ABCDK_TORCH_TAG_CUDA);

    ctx = (abcdk_opencv_stitcher_t *)abcdk_heap_alloc(sizeof(abcdk_opencv_stitcher_t));
    if (!ctx)
        return NULL;

    ctx->tag = tag;

    if (tag == ABCDK_TORCH_TAG_HOST)
    {
        ctx->impl_ctx = new abcdk::opencv::stitcher_cpu();
        if (!ctx->impl_ctx)
            goto ERR;
    }
    else if (tag == ABCDK_TORCH_TAG_CUDA)
    {
        ctx->impl_ctx = new abcdk::opencv::stitcher_cuda(abcdk_cuda_ctx_getspecific());
        if (!ctx->impl_ctx)
            goto ERR;
    }

    return ctx;

ERR:

    abcdk_opencv_stitcher_destroy(&ctx);
    return NULL;
}

abcdk_object_t *abcdk_opencv_stitcher_metadata_dump(abcdk_opencv_stitcher_t *ctx, const char *magic)
{
    abcdk_object_t *out = NULL;
    std::string out_data;
    int chk;

    assert(ctx != NULL && magic != NULL);

    chk = abcdk::opencv::stitcher::Dump(out_data, *ctx->impl_ctx, magic);
    if (chk != 0)
        return NULL;

    out = abcdk_object_copyfrom(out_data.c_str(), out_data.length());
    if (!out)
        return NULL;

    return out;
}

int abcdk_opencv_stitcher_metadata_load(abcdk_opencv_stitcher_t *ctx, const char *magic, const char *data)
{
    int chk;

    assert(ctx != NULL && magic != NULL && data != NULL);

    chk = abcdk::opencv::stitcher::Load(data, *ctx->impl_ctx, magic);
    if (chk == 0)
        return 0;
    else if (chk == -127)
        return -127;

    return -1;
}

int abcdk_opencv_stitcher_set_feature_finder(abcdk_opencv_stitcher_t *ctx, const char *name)
{
    int chk;

    assert(ctx != NULL && name != NULL);

    chk = ctx->impl_ctx->set_feature_finder(name);
    if(chk != 0)
        return -1;

    return 0;
}

int abcdk_opencv_stitcher_estimate_transform(abcdk_opencv_stitcher_t *ctx, int count, abcdk_torch_image_t *img[], abcdk_torch_image_t *mask[], float good_threshold)
{
    std::vector<cv::Mat> tmp_imgs;
    std::vector<cv::Mat> tmp_masks;
    int src_depth;
    int chk;

    assert(ctx != NULL && count >= 2 && img != NULL && mask != NULL);

    tmp_imgs.resize(count);
    tmp_masks.resize(count);

    for (int i = 0; i < count; i++)
    {
        auto &dst_img = tmp_imgs[i];
        auto &dst_mask = tmp_masks[i];
        auto &src_img = img[i];
        auto &src_mask = mask[i];

        assert(src_img != NULL);
        assert(src_img->tag == ABCDK_TORCH_TAG_HOST);
        assert(src_img->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
               src_img->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
               src_img->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
               src_img->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
               src_img->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

        src_depth = abcdk_torch_pixfmt_channels(src_img->pixfmt);

        /*用已存在数据构造cv::Mat对象。*/
        dst_img = cv::Mat(src_img->height, src_img->width, CV_8UC(src_depth), (void *)src_img->data[0], src_img->stride[0]);

        if (src_mask)
        {
            assert(src_mask->tag == ABCDK_TORCH_TAG_HOST);
            assert(src_mask->pixfmt == ABCDK_TORCH_PIXFMT_GRAY8 ||
                   src_mask->pixfmt == ABCDK_TORCH_PIXFMT_RGB24 ||
                   src_mask->pixfmt == ABCDK_TORCH_PIXFMT_BGR24 ||
                   src_mask->pixfmt == ABCDK_TORCH_PIXFMT_RGB32 ||
                   src_mask->pixfmt == ABCDK_TORCH_PIXFMT_BGR32);

            src_depth = abcdk_torch_pixfmt_channels(src_mask->pixfmt);

            /*用已存在数据构造cv::Mat对象。*/
            dst_mask = cv::Mat(src_mask->height, src_mask->width, CV_8UC(src_depth), (void *)src_mask->data[0], src_mask->stride[0]);
        }
    }

    chk = ctx->impl_ctx->EstimateTransform(tmp_imgs, tmp_masks, good_threshold);
    if (chk != 0)
        return -1;

    return 0;
}

int abcdk_opencv_stitcher_set_warper(abcdk_opencv_stitcher_t *ctx,const char *name)
{
    int chk;

    assert(ctx != NULL && name != NULL);

    chk = ctx->impl_ctx->set_warper(name);
    if(chk != 0)
        return -1;

    return 0;
}

int abcdk_opencv_stitcher_build_panorama_param(abcdk_opencv_stitcher_t *ctx)
{
    int chk;

    assert(ctx != NULL);

    chk = ctx->impl_ctx->BuildPanoramaParam();
    if(chk != 0)
        return -1;

    return 0;
}

int abcdk_opencv_stitcher_compose_panorama(abcdk_opencv_stitcher_t *ctx, abcdk_torch_image_t *out, int count, abcdk_torch_image_t *img[])
{
    std::vector<abcdk_torch_image_t *> tmp_imgs;
    int chk;

    assert(ctx != NULL && out != NULL && count >= 2 && img != NULL);

    assert(out->tag == ctx->tag);

    tmp_imgs.resize(count);

    for (int i = 0; i < count; i++)
    {
        abcdk_torch_image_t *img_it = img[i];

        assert(img_it != NULL);
        assert(img_it->tag == ctx->tag);

        tmp_imgs[i] = img_it;
    }

    chk = ctx->impl_ctx->ComposePanorama<abcdk_torch_image_t>(out, tmp_imgs);
    if (chk != 0)
        return -1;

    return 0;
}

#else // OPENCV_STITCHING_STITCHER_HPP

void abcdk_opencv_stitcher_destroy(abcdk_opencv_stitcher_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
}

abcdk_opencv_stitcher_t *abcdk_opencv_stitcher_create()
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return NULL;
}

abcdk_opencv_stitcher_t *abcdk_opencv_stitcher_create_cuda(CUcontext cuda_ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return NULL;
}

abcdk_object_t *abcdk_opencv_stitcher_metadata_dump(abcdk_opencv_stitcher_t *ctx, const char *magic)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return NULL;
}

int abcdk_opencv_stitcher_metadata_load(abcdk_opencv_stitcher_t *ctx, const char *magic, const char *data)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

void abcdk_opencv_stitcher_set_feature_finder(abcdk_opencv_stitcher_t *ctx, const char *name)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return ;
}

int abcdk_opencv_stitcher_estimate_transform(abcdk_opencv_stitcher_t *ctx, int count, abcdk_torch_image_t *img[], abcdk_torch_image_t *mask[], float good_threshold)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

void abcdk_opencv_stitcher_set_warper(abcdk_opencv_stitcher_t *ctx,const char *name)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return ;
}

int abcdk_opencv_stitcher_build_panorama_param(abcdk_opencv_stitcher_t *ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

int abcdk_opencv_stitcher_compose_panorama(abcdk_opencv_stitcher_t *ctx, abcdk_torch_image_t *out, int count, abcdk_torch_image_t *img[])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

#endif // OPENCV_STITCHING_STITCHER_HPP


__END_DECLS
