/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"
#include "abcdk/util/trace.h"
#include "jenc.hxx"

#ifdef __x86_64__

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace jenc
        {
            typedef struct _metadata
            {
                cudaStream_t cu_stream;
                nvjpegHandle_t cu_ctx;
                nvjpegEncoderState_t cu_state;
                nvjpegEncoderParams_t cu_params;
            } metadata_t;

            void free(metadata_t **ctx)
            {
                metadata_t *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                if (ctx_p->cu_params)
                    nvjpegEncoderParamsDestroy(ctx_p->cu_params);
                if (ctx_p->cu_state)
                    nvjpegEncoderStateDestroy(ctx_p->cu_state);
                if (ctx_p->cu_ctx)
                    nvjpegDestroy(ctx_p->cu_ctx);
                if (ctx_p->cu_stream)
                    cudaStreamDestroy(ctx_p->cu_stream);

                delete ctx_p;
            }

            metadata_t *alloc()
            {
                metadata_t *ctx;
                cudaError_t cu_chk;
                nvjpegStatus_t jpeg_chk;

                ctx = new metadata_t;
                if (!ctx)
                    return NULL;

                ctx->cu_ctx = NULL;
                ctx->cu_params = NULL;
                ctx->cu_state = NULL;
                ctx->cu_stream = NULL;

                cu_chk = cudaStreamCreateWithFlags(&ctx->cu_stream, cudaStreamDefault);
                if (cu_chk != cudaSuccess)
                {
                    free(&ctx);
                    return NULL;
                }

                jpeg_chk = nvjpegCreateSimple(&ctx->cu_ctx);
                if (jpeg_chk != NVJPEG_STATUS_SUCCESS)
                {
                    free(&ctx);
                    return NULL;
                }

                nvjpegEncoderStateCreate(ctx->cu_ctx, &ctx->cu_state, ctx->cu_stream);
                nvjpegEncoderParamsCreate(ctx->cu_ctx, &ctx->cu_params, ctx->cu_stream);
                nvjpegEncoderParamsSetQuality(ctx->cu_params, 100, ctx->cu_stream);
                nvjpegEncoderParamsSetSamplingFactors(ctx->cu_params, NVJPEG_CSS_420, ctx->cu_stream);

                return ctx;
            }

            abcdk_object_t *encode(metadata_t *ctx, const image::metadata_t *src)
            {
                image::metadata_t *tmp_src;
                abcdk_object_t *dst;
                size_t dst_size = 0;
                nvjpegImage_t src_data = {0};
                nvjpegStatus_t jpeg_chk;
                cudaError_t cu_chk;
                int chk;

                if (src->format != AV_PIX_FMT_RGB24 && src->format != AV_PIX_FMT_BGR24)
                {
                    tmp_src = image::create(src->width,src->height, ABCDK_XPU_PIXFMT_BGR24 ,16,0);
                    if(!tmp_src)
                        return NULL;

                    chk = imgproc::convert(src,tmp_src);
                    if(chk != 0)
                    {
                        image::free(&tmp_src);
                        return NULL;
                    }

                    dst = encode(ctx, tmp_src);
                    image::free(&tmp_src);

                    return dst;
                }

                src_data.channel[0] = src->data[0];
                src_data.pitch[0] = src->linesize[0];

                if (src->format == AV_PIX_FMT_RGB24)
                    jpeg_chk = nvjpegEncodeImage(ctx->cu_ctx, ctx->cu_state, ctx->cu_params, &src_data, NVJPEG_INPUT_RGBI, src->width, src->height, ctx->cu_stream);
                else if (src->format == AV_PIX_FMT_BGR24)
                    jpeg_chk = nvjpegEncodeImage(ctx->cu_ctx, ctx->cu_state, ctx->cu_params, &src_data, NVJPEG_INPUT_BGRI, src->width, src->height, ctx->cu_stream);

                if (jpeg_chk != NVJPEG_STATUS_SUCCESS)
                    return NULL;

                jpeg_chk = nvjpegEncodeRetrieveBitstream(ctx->cu_ctx, ctx->cu_state, NULL, &dst_size, ctx->cu_stream);
                if (jpeg_chk != NVJPEG_STATUS_SUCCESS)
                    return NULL;

                cu_chk = cudaStreamSynchronize(ctx->cu_stream);
                if (cu_chk != cudaSuccess)
                    return NULL;

                dst = abcdk_object_alloc2(dst_size);
                if (!dst)
                    return NULL;

                jpeg_chk = nvjpegEncodeRetrieveBitstream(ctx->cu_ctx, ctx->cu_state, dst->pptrs[0], &dst_size, ctx->cu_stream);
                if (jpeg_chk != NVJPEG_STATUS_SUCCESS)
                {
                    abcdk_object_unref(&dst);
                    return NULL;
                }

                return dst;
            }

        } // namespace jenc
    } // namespace nvidia

} // namespace abcdk_xpu

#endif //#ifdef __x86_64__