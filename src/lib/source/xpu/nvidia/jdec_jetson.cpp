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

#ifdef __XPU_NVIDIA__MMAPI__

#include "NvJpegDecoder.h"
#include "NvBufSurface.h"

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

                ctx->cu_ctx = NULL;

                return ctx;
            }

            image::metadata_t *decode(metadata_t *ctx, const void *src, int src_size)
            {
                NvBufSurf::NvCommonTransformParams transform_params;
                NvBufSurf::NvCommonAllocateParams params;
                int dma_fd = -1;
                int cp_dma_fd = -1;
                uint32_t pixfmt = -1;
                uint32_t width = 0;
                uint32_t height = 0;
                NvBufSurface *cp_dma_surf = NULL;
                image::metadata_t *dst = NULL;
                int chk;

                if(!ctx->cu_ctx)
                {
                    ctx->cu_ctx = NvJPEGDecoder::createJPEGDecoder("nvjpegdec");
                    if(!ctx->cu_ctx)
                        return NULL;
                }

                chk = ctx->cu_ctx->decodeToFd(dma_fd, (uint8_t *)src, src_size, pixfmt, width, height);
                if (chk != 0)
                    return NULL;

                
                params.memType = NVBUF_MEM_SURFACE_ARRAY;
                params.width = width;
                params.height = height;
                params.layout = NVBUF_LAYOUT_PITCH;
                params.colorFormat = NVBUF_COLOR_FORMAT_YUV420;
                params.memtag = NvBufSurfaceTag_VIDEO_CONVERT;

                chk = NvBufSurf::NvAllocate(&params, 1, &cp_dma_fd);
                if (chk != 0)
                    return NULL;
                
                transform_params.src_top = 0;
                transform_params.src_left = 0;
                transform_params.src_width = width;
                transform_params.src_height = height;
                transform_params.dst_top = 0;
                transform_params.dst_left = 0;
                transform_params.dst_width = width;
                transform_params.dst_height = height;
                transform_params.flag = NVBUFSURF_TRANSFORM_FILTER;
                transform_params.flip = NvBufSurfTransform_None;
                transform_params.filter = NvBufSurfTransformInter_Nearest;
                chk = NvBufSurf::NvTransform(&transform_params, dma_fd, cp_dma_fd);
                if(chk != 0)
                    return NULL;
                    
                chk = NvBufSurfaceFromFd(cp_dma_fd, (void **)&cp_dma_surf);
                if (chk != 0)
                {
                    NvBufSurf::NvDestroy(cp_dma_fd);
                    return NULL;
                }

                chk = image::reset(&dst, width, height, ABCDK_XPU_PIXFMT_YUV420P, 16, 0);
                if (chk != 0)
                {
                    NvBufSurf::NvDestroy(cp_dma_fd);
                    return NULL;
                }

                chk = image::copy(cp_dma_surf, 1, dst, 0);
                NvBufSurf::NvDestroy(cp_dma_fd);

                if (chk != 0)
                    return NULL;

                return dst;
            }

        } // namespace jdec
    } // namespace nvidia
} // namespace abcdk_xpu

#endif // #ifdef __XPU_NVIDIA__MMAPI__