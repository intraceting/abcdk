/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "jenc.hxx"

#ifdef __aarch64__

#include "NvJpegEncoder.h"
#include "NvBufSurface.h"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace jenc
        {
            typedef struct _metadata
            {
                NvJPEGEncoder *cu_ctx;
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

                ctx->cu_ctx = NvJPEGEncoder::createJPEGEncoder("jpenenc");

                return ctx;
            }

            abcdk_object_t *encode(metadata_t *ctx, const image::metadata_t *src)
            {
                image::metadata_t *tmp_src;
                int dma_fd = -1;
                NvBufSurface *dma_surf = NULL;
                abcdk_object_t *dst;
                int chk;

                if (src->format != AV_PIX_FMT_YUV420P)
                {
                    tmp_src = image::create(src->width,src->height, ABCDK_XPU_PIXFMT_YUV420P ,16,0);
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

                NvBufSurf::NvCommonAllocateParams params;

                params.memType = NVBUF_MEM_SURFACE_ARRAY;
                params.width = src->width;
                params.height = src->height;
                params.layout = NVBUF_LAYOUT_PITCH;
                params.colorFormat = NVBUF_COLOR_FORMAT_YUV420;
                params.memtag = NvBufSurfaceTag_VIDEO_CONVERT;

                chk = NvBufSurf::NvAllocate(&params, 1, &dma_fd);
                if(chk != 0)
                    return NULL;

                chk = NvBufSurfaceFromFd(dma_fd, (void **)&dma_surf);
                if (chk != 0)
                {
                    NvBufSurf::NvDestroy(dma_fd);
                    return NULL;
                }

                for (int i = 0; i < 4; i++)
                {
                    if (src->linesize[i] <= 0)
                        break;

                    NvBufSurfaceMap(dma_surf, 0, i, NVBUF_MAP_READ);
                    NvBufSurfaceSyncForCpu(dma_surf, 0, i);

                    chk = image::copy(src, i, 0, dma_surf->surfaceList[0].mappedAddr.addr[i], dma_surf->surfaceList[0].pitch, 1);
                    assert(chk == 0);

                    NvBufSurfaceUnMap(dma_surf, 0, i);
                }

                dst = abcdk_object_alloc2(src->width * src->height * 3 / 2);
                if(!dst)
                {
                    NvBufSurfaceDestroy(dma_surf);
                    return NULL;
                }

                chk = ctx->cu_ctx->encodeFromFd(dma_fd, JCS_YCbCr, &dst->pptrs[0], dst->sizes[0], 100);
                NvBufSurfaceDestroy(dma_surf);
                
                if(chk != 0)
                    return NULL;

                return dst;
            }

        } // namespace jenc
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // #ifdef __aarch64__