/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "jenc.hxx"

#ifdef __aarch64__

#include "jetson/NvJpegDecoder.h"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace jdec
        {
            typedef struct _metadata
            {
                NvJPEGDecoder *cu_ctx;
            } metadata_t;

            void free(metadata_t **ctx)
            {
                metadata_t *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                common::util::delete_object(&ctx_p->cu_ctx);

                common::util::delete_object(&ctx_p);
            }

            metadata_t *alloc()
            {
                metadata_t *ctx;

                ctx = new metadata_t;
                if (!ctx)
                    return NULL;

                ctx->cu_ctx = NvJPEGDecoder::createJPEGDecoder("jpegdec");


                return ctx;
            }

            image::metadata_t *decode(metadata_t *ctx, const void *src, int src_size)
            {
                NvBuffer *buffer = NULL;
                uint32_t pixfmt = -1;
                uint32_t width = 0;
                uint32_t height = 0;
                image::metadata_t *dst = NULL;
                uint8_t *src_data[4] = {NULL, NULL, NULL, NULL};
                int src_linesize[4] = {-1, -1, -1, -1};
                int chk;

                chk = ctx->cu_ctx->decodeToBuffer(&buffer, (uint8_t*)src, src_size, &pixfmt, &width, &height);
                if(chk != 0)
                    return NULL;

                if (pixfmt == V4L2_PIX_FMT_YUV422M)
                    chk = image::reset(&dst,width,height,ABCDK_XPU_PIXFMT_YUV422P,16,0);
                else if (pixfmt == V4L2_PIX_FMT_YUV420M)
                    chk = image::reset(&dst,width,height,ABCDK_XPU_PIXFMT_YUV420P,16,0);
                else if (pixfmt == V4L2_PIX_FMT_YUV444M)
                    chk = image::reset(&dst,width,height,ABCDK_XPU_PIXFMT_YUV444P,16,0);
                else if (pixfmt == V4L2_PIX_FMT_YUV422RM)
                    chk = image::reset(&dst,width,height,ABCDK_XPU_PIXFMT_YUV420P,16,0);
                else 
                {
                    common::util::delete_object(&buffer);
                    return NULL;
                }

                for (int i = 0; i < buffer->n_planes; i++)
                {
                    src_data[i] = buffer->planes->data;
                    src_linesize[i] = buffer->planes->fmt.stride;
                }

                image::upload((const uint8_t**)src_data,src_linesize,dst);
                common::util::delete_object(&buffer);

                return dst;
            }
        } // namespace jdec
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // #ifdef __aarch64__