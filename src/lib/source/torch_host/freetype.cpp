/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/freetype.h"

__BEGIN_DECLS

#ifdef _OPENCV_FREETYPE_H_

/**简单的文字引擎。*/
struct _abcdk_torch_freetype
{   
    /**引擎。*/
    cv::Ptr<cv::freetype::FreeType2> impl_ctx;

};// abcdk_torch_freetype_t;


void abcdk_torch_freetype_destroy(abcdk_torch_freetype_t **ctx)
{
    abcdk_torch_freetype_t *ctx_p;

    if(!ctx ||!*ctx)
        return;

    ctx_p = *ctx;
    *ctx = NULL;

    /*覆盖智能指针等同于释放对象。*/
    ctx_p->impl_ctx = NULL;

    abcdk_heap_free(ctx_p);
}


abcdk_torch_freetype_t *abcdk_torch_freetype_create()
{
    abcdk_torch_freetype_t *ctx;

    ctx = (abcdk_torch_freetype_t *)abcdk_heap_alloc(sizeof(abcdk_torch_freetype_t));
    if (!ctx)
        return NULL;

    ctx->impl_ctx = cv::freetype::createFreeType2();
    if(!ctx->impl_ctx)
        goto ERR;

    return ctx;

ERR:

    abcdk_torch_freetype_destroy(&ctx);
    return NULL;
}

int abcdk_torch_freetype_load_font(abcdk_torch_freetype_t *ctx, const char *file, int id)
{
    assert(ctx != NULL && file != NULL && id >= 0);

    try
    {
        ctx->impl_ctx->loadFontData(file, id);
    }
    catch (const cv::Exception &e)
    {
        abcdk_trace_printf(LOG_WARNING, "%s(errno=%d).", e.err.c_str(), e.code);
        return -1;
    }

    return 0;
}

int abcdk_torch_freetype_set_split_number(abcdk_torch_freetype_t *ctx, int num)
{
    assert(ctx != NULL && num >=0);

    try
    {
        ctx->impl_ctx->setSplitNumber(num);
    }
    catch (const cv::Exception &e)
    {
        abcdk_trace_printf(LOG_WARNING, "%s(errno=%d).", e.err.c_str(), e.code);
        return -1;
    }

    return 0;
}

int abcdk_torch_freetype_get_text_size(abcdk_torch_freetype_t *ctx,
                                        abcdk_torch_size_t *size, const char *text,
                                        int height, int thickness, int *base_line)
{
    assert(ctx != NULL && size != NULL && text != NULL && height >=0);

    try
    {
        cv::Size tmp_size = ctx->impl_ctx->getTextSize(text,height,thickness,base_line);

        size->width = tmp_size.width;
        size->height = tmp_size.height;
    }
    catch (const cv::Exception &e)
    {
        abcdk_trace_printf(LOG_WARNING, "%s(errno=%d).", e.err.c_str(), e.code);
        return -1;
    }

    return 0;
}

int abcdk_torch_freetype_put_text_host(abcdk_torch_freetype_t *ctx,
                                   abcdk_torch_image_t *img, const char *text,
                                   abcdk_torch_point_t *org, int height, uint32_t color[4],
                                   int thickness, int line_type, uint8_t bottom_left_origin)
{
    assert(ctx != NULL && img != NULL && text != NULL && org != NULL && height >=0 && line_type >= 0 && color != NULL);


    try
    {
        int depth = abcdk_torch_pixfmt_channels(img->pixfmt);
        cv::Mat tmp_img = cv::Mat(img->height,img->width,CV_8UC(depth),(void*)img->data[0],img->stride[0]);

        ctx->impl_ctx->putText(tmp_img, text, cv::Point(org->x,org->y),height, cv::Scalar(color[0], color[1], color[2]), thickness, line_type, bottom_left_origin);
    }
    catch (const cv::Exception &e)
    {
        abcdk_trace_printf(LOG_WARNING, "%s(errno=%d).", e.err.c_str(), e.code);
        return -1;
    }

    return 0;
}

#else //_OPENCV_FREETYPE_H_

void abcdk_torch_freetype_destroy(abcdk_torch_freetype_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return ;
}

abcdk_torch_freetype_t *abcdk_torch_freetype_create()
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return NULL;
}

int abcdk_torch_freetype_load_font(abcdk_torch_freetype_t *ctx, const char *file,int id)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

int abcdk_torch_freetype_set_split_number(abcdk_torch_freetype_t *ctx, int num)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

int abcdk_torch_freetype_get_text_size(abcdk_torch_freetype_t *ctx,
                                        abcdk_torch_size_t *size, const char *text,
                                        int height, int thickness, int *base_line)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

int abcdk_torch_freetype_put_text_host(abcdk_torch_freetype_t *ctx,
                                   abcdk_torch_image_t *img, const char *text,
                                   abcdk_torch_point_t *org, int height, uint32_t color[4],
                                   int thickness, int line_type, uint8_t bottom_left_origin)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

#endif // _OPENCV_FREETYPE_H_


__END_DECLS
