/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/jcodec.h"
#include "abcdk/torch/opencv.h"


__BEGIN_DECLS

#ifdef OPENCV_IMGCODECS_HPP

/** JPEG编/解码器。*/
typedef struct _abcdk_torch_jcodec_host
{
    /**编码器。!0 是，0 否。*/
    int encoder;

    /**
     * 质量。
     *
     * 1~99 值越大越清晰，占用的空间越多。
     */
    int quality;

} abcdk_torch_jcodec_host_t;

void abcdk_torch_jcodec_free_host(abcdk_torch_jcodec_t **ctx)
{
    abcdk_torch_jcodec_t *ctx_p;
    abcdk_torch_jcodec_host_t *ht_ctx_p;

    if(!ctx || !*ctx)
        return ;

    ctx_p = *ctx;
    *ctx = NULL;

    assert(ctx_p->tag == ABCDK_TORCH_TAG_HOST);

    ht_ctx_p = (abcdk_torch_jcodec_host_t *)ctx_p->private_ctx;

    abcdk_heap_free(ht_ctx_p);
    abcdk_heap_free(ctx_p);
}

abcdk_torch_jcodec_t *abcdk_torch_jcodec_alloc_host(int encoder)
{
    abcdk_torch_jcodec_t *ctx;
    abcdk_torch_jcodec_host_t *ht_ctx_p;

    ctx = (abcdk_torch_jcodec_t *)abcdk_heap_alloc(sizeof(abcdk_torch_jcodec_t));
    if(!ctx)
        return NULL;

    ctx->tag = ABCDK_TORCH_TAG_HOST;

    ctx->private_ctx = abcdk_heap_alloc(sizeof(abcdk_torch_jcodec_host_t));
    if(!ctx->private_ctx)
        goto ERR;

    ht_ctx_p = (abcdk_torch_jcodec_host_t *)ctx->private_ctx;

    ht_ctx_p->encoder = encoder;

    return ctx;

ERR:

    abcdk_torch_jcodec_free_host(&ctx);
    return NULL;
}

int abcdk_torch_jcodec_start_host(abcdk_torch_jcodec_t *ctx, abcdk_torch_jcodec_param_t *param)
{
    abcdk_torch_jcodec_host_t *ht_ctx_p;
    int chk;

    assert(ctx != NULL && param != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

    ht_ctx_p = (abcdk_torch_jcodec_host_t *)ctx->private_ctx;

    ht_ctx_p->quality = param->quality;

    /*修正到合理区间。*/
    ht_ctx_p->quality = ABCDK_CLAMP(ht_ctx_p->quality,1,99);

    return 0;
}

abcdk_object_t *abcdk_torch_jcodec_encode_host(abcdk_torch_jcodec_t *ctx, const abcdk_torch_image_t *src)
{
    abcdk_torch_jcodec_host_t *ht_ctx_p;
    abcdk_torch_image_t *tmp_src = NULL;
    abcdk_object_t *dst = NULL;
    int chk;

    assert(ctx != NULL && src != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);
    assert(src->tag == ABCDK_TORCH_TAG_HOST);
        
    ht_ctx_p = (abcdk_torch_jcodec_host_t *)ctx->private_ctx;

    if (src->pixfmt != ABCDK_TORCH_PIXFMT_BGR24)
    {
        tmp_src = abcdk_torch_image_create_host(src->width, src->height, ABCDK_TORCH_PIXFMT_BGR24, 4);
        if (!tmp_src)
            return NULL;

        chk = abcdk_torch_image_convert_host(tmp_src, src);

        /*转格式成功后继续执行保存操作。*/
        if (chk == 0)
            dst = abcdk_torch_jcodec_encode_host(ctx, tmp_src);

        abcdk_torch_image_free_host(&tmp_src);
        return dst;
    }

    std::vector<uint8_t> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, ht_ctx_p->quality};

    /*用已存在数据构造cv::Mat对象。*/
    cv::Mat img = cv::Mat(src->height, src->width, CV_8UC3, (void *)src->data[0], src->stride[0]);

    cv::imencode(".jpg", img, buf, params);

    if (buf.size() <= 0)
        return NULL;

    return abcdk_object_copyfrom(buf.data(), buf.size());
}

int abcdk_torch_jcodec_encode_to_file_host(abcdk_torch_jcodec_t *ctx, const char *dst, const abcdk_torch_image_t *src)
{
    abcdk_object_t *tmp_dst = NULL;
    int chk;

    assert(ctx != NULL && dst != NULL && src != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

    tmp_dst = abcdk_torch_jcodec_encode_host(ctx, src);
    if (!tmp_dst)
        return -1;

    if (access(dst, F_OK) == 0)
    {
        chk = truncate(dst, 0);
        if (chk != 0)
        {
            abcdk_object_unref(&tmp_dst);
            return -1;
        }
    }

    chk = abcdk_save(dst, tmp_dst->pptrs[0], tmp_dst->sizes[0], 0);
    if (chk != tmp_dst->sizes[0])
    {
        abcdk_object_unref(&tmp_dst);
        return -1;
    }

    abcdk_object_unref(&tmp_dst);
    return 0;
}

abcdk_torch_image_t *abcdk_torch_jcodec_decode_host(abcdk_torch_jcodec_t *ctx, const void *src, int src_size)
{
    abcdk_torch_image_t *dst = NULL;
    int chk;

    assert(ctx != NULL && src != NULL && src_size > 0);
    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

    std::vector<uint8_t> buf_src(ABCDK_PTR2U8PTR(src, 0), ABCDK_PTR2U8PTR(src, src_size));

    cv::Mat tmp_src = cv::imdecode(buf_src, cv::IMREAD_COLOR);
    if (tmp_src.empty())
        return NULL;

    dst = abcdk_torch_image_create_host(tmp_src.cols, tmp_src.rows, ABCDK_TORCH_PIXFMT_BGR24, 1);
    if (!dst)
        return NULL;

    abcdk_torch_image_copy_plane_host(dst, 0, tmp_src.data, tmp_src.step);

    return dst;
}

abcdk_torch_image_t *abcdk_torch_jcodec_decode_from_file_host(abcdk_torch_jcodec_t *ctx, const char *src)
{
    abcdk_torch_image_t *dst = NULL;
    int chk;

    assert(ctx != NULL && src != NULL);
    assert(ctx->tag == ABCDK_TORCH_TAG_HOST);

    cv::Mat tmp_src = cv::imread(src, cv::IMREAD_COLOR);
    if (tmp_src.empty())
        return NULL;

    dst = abcdk_torch_image_create_host(tmp_src.cols, tmp_src.rows, ABCDK_TORCH_PIXFMT_BGR24, 1);
    if (!dst)
        return NULL;

    abcdk_torch_image_copy_plane_host(dst, 0, tmp_src.data, tmp_src.step);

    return dst;
}

#else //OPENCV_IMGCODECS_HPP

void abcdk_torch_jcodec_free_host(abcdk_torch_jcodec_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return ;
}

abcdk_torch_jcodec_t *abcdk_torch_jcodec_alloc_host(int encoder)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return NULL;
}

int abcdk_torch_jcodec_start_host(abcdk_torch_jcodec_t *ctx, abcdk_torch_jcodec_param_t *param)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

abcdk_object_t *abcdk_torch_jcodec_encode_host(abcdk_torch_jcodec_t *ctx, const abcdk_torch_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return NULL;
}

int abcdk_torch_jcodec_encode_to_file_host(abcdk_torch_jcodec_t *ctx, const char *dst, const abcdk_torch_image_t *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

abcdk_torch_image_t *abcdk_torch_jcodec_decode_host(abcdk_torch_jcodec_t *ctx, const void *src, int src_size)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return NULL;
}

abcdk_torch_image_t *abcdk_torch_jcodec_decode_from_file_host(abcdk_torch_jcodec_t *ctx, const char *src)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return NULL;
}

#endif //OPENCV_IMGCODECS_HPP


__END_DECLS