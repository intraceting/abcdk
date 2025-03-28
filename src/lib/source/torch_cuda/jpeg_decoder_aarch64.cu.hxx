/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_JPEG_DECODER_AARCH64_HXX
#define ABCDK_TORCH_NVIDIA_JPEG_DECODER_AARCH64_HXX

#include "abcdk/util/option.h"
#include "abcdk/torch/nvidia.h"
#include "abcdk/torch/image.h"
#include "jpeg_decoder.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef __aarch64__

namespace abcdk
{
    namespace cuda
    {
        namespace jpeg
        {
            class decoder_aarch64 : public decoder
            {
            public:
                static decoder *create(CUcontext cuda_ctx)
                {
                    decoder *ctx = new decoder_aarch64(cuda_ctx);
                    if (!ctx)
                        return NULL;

                    return ctx;
                }

                static void destory(decoder **ctx)
                {
                    decoder *ctx_p;

                    if (!ctx || !*ctx)
                        return;

                    ctx_p = *ctx;
                    *ctx = NULL;

                    delete (decoder_aarch64 *)ctx_p;
                }

            private:
                CUcontext m_gpu_ctx;

                NvJPEGDecoder *m_ctx;

            public:
                decoder_aarch64(CUcontext cuda_ctx)
                {
                    m_gpu_ctx = cuda_ctx;

                    m_ctx = NULL;
                }

                virtual ~decoder_aarch64()
                {
                    close();
                }

            public:
                virtual void close()
                {
                    if (m_gpu_ctx)
                        cuCtxPushCurrent(m_gpu_ctx);

                    if (m_ctx)
                        delete m_ctx;
                    m_ctx = NULL;

                    if (m_gpu_ctx)
                        cuCtxPopCurrent(NULL);
                }

                virtual int open(abcdk_torch_jcodec_param_t *param)
                {
                    cudaError_t cuda_chk;

                    assert(param != NULL);

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    m_ctx = NvJPEGDecoder::createJPEGDecoder("jpegdec");
                    if (!m_ctx)
                        return -1;

                    return 0;
                }

                virtual abcdk_torch_image_t *update(const void *src, int src_size)
                {
                    abcdk_torch_image_t *dst;
                    NvBuffer *buffer = NULL;
                    uint32_t pixfmt = 0, width = 0, height = 0;
                    int chk;

                    assert(src != NULL && src_size > 0);

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    chk = m_ctx->decodeToBuffer(&buffer, (uint8_t *)src, src_size, &pixfmt, &width, &height);
                    if (chk != 0)
                        return NULL;

                    if (pixfmt == V4L2_PIX_FMT_YUV420M)
                        dst = abcdk_torch_frame_create_cuda(width, height, ABCDK_TORCH_PIXFMT_YUV420P, 1);
                    else if (pixfmt == V4L2_PIX_FMT_YUV422M)
                        dst = abcdk_torch_frame_create_cuda(width, height, ABCDK_TORCH_PIXFMT_YUV422P, 1);
                    else if (pixfmt == V4L2_PIX_FMT_YUV444M)
                        dst = abcdk_torch_frame_create_cuda(width, height, ABCDK_TORCH_PIXFMT_YUV444P, 1);
                    else
                        return NULL;

                    if (!dst)
                        return NULL;

                    for (int i = 0; i < buffer->n_planes; i++)
                    {
                        NvBuffer::NvBufferPlane &plane = buffer->planes[i];

                        abcdk_torch_memcpy_2d_cuda(dst->data[i], dst->linesize[i], 0, 0, 0,
                                                   plane.data, plane.fmt.stride, 0, 0, 0,
                                                   plane.fmt.bytesperpixel * plane.fmt.width, plane.fmt.height);
                    }

                    return dst;
                }

                virtual abcdk_torch_image_t *update(const void *src)
                {
                    abcdk_object_t *src_data;
                    abcdk_torch_image_t *dst;

                    assert(src != NULL);

                    src_data = abcdk_object_copyfrom_file(src);
                    if (!src_data)
                        return NULL;

                    dst = update(src_data->pptrs[0], src_data->sizes[0]);
                    abcdk_object_unref(&src_data);

                    return dst;
                }
            };
        } // namespace jpeg
    } // namespace cuda
} // namespace abcdk

#endif //__aarch64__
#endif // __cuda_cuda_h__

#endif // ABCDK_TORCH_NVIDIA_JPEG_DECODER_AARCH64_HXX