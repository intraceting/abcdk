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


                return NULL;
            }

            image::metadata_t *decode(metadata_t *ctx, const void *src, int src_size)
            {
                NvBuffer *buffer = NULL;
                uint32_t pixfmt = -1;
                uint32_t width = 0;
                uint32_t height = 0;
                int chk;

                chk = ctx->cu_ctx->decodeToBuffer(&buffer, (uint8_t*)src, src_size, &pixfmt, &width, &height);
                if(chk != 0)
                    return NULL;

                

                return NULL;
            }
        } // namespace jdec
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // #ifdef __aarch64__