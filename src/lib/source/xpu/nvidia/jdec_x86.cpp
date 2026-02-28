/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "jdec.hxx"

#ifdef __x86_64__

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace jdec
        {
            typedef struct _metadata
            {
                cudaStream_t cu_stream;
                nvjpegHandle_t cu_ctx;
                nvjpegJpegState_t cu_state;
            } metadata_t;

            void free(metadata_t **ctx)
            {
                metadata_t *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                if (ctx_p->cu_state)
                    nvjpegJpegStateDestroy(ctx_p->cu_state);
                if (ctx_p->cu_ctx)
                    nvjpegDestroy(ctx_p->cu_ctx);

                if (ctx_p->cu_stream)
                    cudaStreamDestroy(ctx_p->cu_stream);

                delete ctx_p;
            }

            static int dev_malloc(void **pptr, size_t size)
            {
                cudaError_t cuda_chk;

                cuda_chk = cudaMalloc(pptr, size);
                if (cuda_chk != cudaSuccess)
                    return -1;

                return 0;
            }

            static int dev_free(void *ptr)
            {
                cudaError_t cuda_chk;

                cuda_chk = cudaFree(ptr);
                if (cuda_chk != cudaSuccess)
                    return -1;

                return 0;
            }

            static int pinned_malloc(void **pptr, size_t size, unsigned int flags)
            {
                cudaError_t cuda_chk;

                cuda_chk = cudaHostAlloc(pptr, size, flags);
                if (cuda_chk != cudaSuccess)
                    return -1;

                return 0;
            }

            static int pinned_free(void *ptr)
            {
                cudaError_t cuda_chk;

                cuda_chk = cudaFreeHost(ptr);
                if (cuda_chk != cudaSuccess)
                    return -1;

                return 0;
            }

            void check_memory_leak_version()
            {
                int full_ver = 0, major_ver = -1, minor_ver = -1;
                cudaError_t chk;

                chk = cudaRuntimeGetVersion(&full_ver);
                if (chk != cudaSuccess)
                    return;

                major_ver = full_ver / 1000;
                minor_ver = (full_ver % 1000) / 10;

                if (major_ver == 11)
                {
                    if (minor_ver >= 0 && minor_ver <= 6)
                        abcdk_trace_printf(LOG_WARNING, ABCDK_GETTEXT("CUDA11.0~CUDA11.6运行库中的nvJPEG解码器存在内存泄漏问题, 谨慎使用."));
                }
            }

            metadata_t *alloc()
            {
                metadata_t *ctx;
                cudaError_t cu_chk;
                nvjpegStatus_t jpeg_chk;
                nvjpegDevAllocator_t dev_allocator = {0};
                nvjpegPinnedAllocator_t pinned_allocator = {0};

                check_memory_leak_version();

                ctx = new metadata_t;
                if (!ctx)
                    return NULL;

                ctx->cu_ctx = NULL;
                ctx->cu_state = NULL;
                ctx->cu_stream = NULL;

                cu_chk = cudaStreamCreateWithFlags(&ctx->cu_stream, cudaStreamDefault);
                if (cu_chk != cudaSuccess)
                {
                    free(&ctx);
                    return NULL;
                }

                dev_allocator.dev_malloc = dev_malloc;
                dev_allocator.dev_free = dev_free;
                pinned_allocator.pinned_malloc = pinned_malloc;
                pinned_allocator.pinned_free = pinned_free;

                jpeg_chk = nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &dev_allocator, &pinned_allocator, NVJPEG_FLAGS_DEFAULT, &ctx->cu_ctx);
                if (jpeg_chk != NVJPEG_STATUS_SUCCESS)
                {
                    free(&ctx);
                    return NULL;
                }

                jpeg_chk = nvjpegJpegStateCreate(ctx->cu_ctx, &ctx->cu_state);
                if (jpeg_chk != NVJPEG_STATUS_SUCCESS)
                {
                    free(&ctx);
                    return NULL;
                }

                return ctx;
            }

            image::metadata_t *decode(metadata_t *ctx, const void *src, int src_size)
            {
                image::metadata_t *dst;
                nvjpegImage_t dst_data = {0};
                int components;
                nvjpegChromaSubsampling_t subsampling;
                int widths[NVJPEG_MAX_COMPONENT] = {0}, heights[NVJPEG_MAX_COMPONENT] = {0};
                nvjpegStatus_t jpeg_chk;
                cudaError_t cu_chk;

                jpeg_chk = nvjpegGetImageInfo(ctx->cu_ctx, (uint8_t *)src, src_size, &components, &subsampling, widths, heights);
                if (jpeg_chk != NVJPEG_STATUS_SUCCESS)
                    return NULL;

                /*全部转成RGB和GRAY.*/
                if (components == 3)
                    dst = image::create(widths[0], heights[0], ABCDK_XPU_PIXFMT_RGB24, 16, 0); // 只使用第一层的宽和高.
                else if (components == 1)
                    dst = image::create(widths[0], heights[0], ABCDK_XPU_PIXFMT_GRAY8, 16, 0);
                else
                    dst = NULL;

                if(!dst)
                    return NULL;

                dst_data.channel[0] = dst->data[0];
                dst_data.pitch[0] = dst->linesize[0];

                if (components == 3)
                    jpeg_chk = nvjpegDecode(ctx->cu_ctx, ctx->cu_state, (uint8_t *)src, src_size, NVJPEG_OUTPUT_RGBI, &dst_data, ctx->cu_stream);
                else if (components == 1)
                    jpeg_chk = nvjpegDecode(ctx->cu_ctx, ctx->cu_state, (uint8_t *)src, src_size, NVJPEG_OUTPUT_Y, &dst_data, ctx->cu_stream);
                else
                    jpeg_chk = NVJPEG_STATUS_JPEG_NOT_SUPPORTED;
               
                if (jpeg_chk != NVJPEG_STATUS_SUCCESS)
                {
                    image::free(&dst);
                    return NULL;
                }

                cu_chk = cudaStreamSynchronize(ctx->cu_stream);
                if (cu_chk != cudaSuccess)
                {
                    image::free(&dst);
                    return NULL;
                }

                return dst;
            }

        } // namespace jdec
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // #ifdef __x86_64__