/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_JPEG_ENCODER_AARCH64_HXX
#define ABCDK_TORCH_NVIDIA_JPEG_ENCODER_AARCH64_HXX

#include "abcdk/util/option.h"
#include "abcdk/torch/image.h"
#include "abcdk/torch/nvidia.h"
#include "jpeg_encoder.hxx"

#ifdef __cuda_cuda_h__
#ifdef __aarch64__

namespace abcdk
{
    namespace torch_cuda
    {
        namespace jpeg
        {
            class encoder_aarch64 : public encoder
            {
            public:
                static encoder *create(CUcontext cuda_ctx)
                {
                    encoder *ctx = new encoder_aarch64(cuda_ctx);
                    if (!ctx)
                        return NULL;

                    return ctx;
                }

                static void destory(encoder **ctx)
                {
                    encoder *ctx_p;

                    if (!ctx || !*ctx)
                        return;

                    ctx_p = *ctx;
                    *ctx = NULL;

                    delete (encoder_aarch64 *)ctx_p;
                }

            private:
                abcdk_option_t *m_cfg;

                CUcontext m_gpu_ctx;

                int m_quality;
                NvJPEGEncoder *m_ctx;

            public:
                encoder_aarch64(CUcontext cuda_ctx)
                {
                    m_cfg = NULL;
                    m_gpu_ctx = cuda_ctx;
                    m_ctx = NULL;
                }

                virtual ~encoder_aarch64()
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

                    abcdk_option_free(&m_cfg);
                }

                virtual int open(abcdk_torch_jcodec_param_t *param)
                {
                    cudaError_t cuda_chk;
                    nvjpegStatus_t jpeg_chk;

                    assert(param != NULL);

                    m_quality = ABCDK_CLAMP(param->quality, 1, 99);

                    abcdk::torch_cuda::context_robot robot(m_gpu_ctx);

                    m_ctx = NvJPEGEncoder::createJPEGEncoder("jpenenc");
                    if (!m_ctx)
                        return -1;

                    return 0;
                }

                virtual abcdk_object_t *update(const abcdk_torch_image_t *src)
                {
                    abcdk_torch_image_t *tmp_src = NULL;
                    abcdk_object_t *dst;
                    unsigned long out_size = 0;
                    int chk;

                    assert(src != NULL);

                    abcdk::torch_cuda::context_robot robot(m_gpu_ctx);

                    if (src->format != ABCDK_TORCH_PIXFMT_YUV420P)
                    {
                        tmp_src = abcdk_torch_frame_create_cuda(src->width, src->height, ABCDK_TORCH_PIXFMT_YUV420P, 1);
                        if (!tmp_src)
                            return NULL;

                        chk = abcdk_torch_frame_convert_cuda(tmp_src, src); // 转换格式。

                        if (chk == 0)
                            dst = update(tmp_src);

                        av_frame_free(&tmp_src);

                        return dst;
                    }

                    NvBuffer buffer(V4L2_PIX_FMT_YUV420M, src->width, src->height, 0);
                    chk = buffer.allocateMemory();
                    if (chk != 0)
                        return NULL;

                    for (int i = 0; i < buffer.n_planes; i++)
                    {
                        NvBuffer::NvBufferPlane &plane = buffer.planes[i];

                        abcdk_torch_memcpy_2d_cuda(plane.data, plane.fmt.stride, 0, 0, 0,
                                                   src->data[i], src->linesize[i], 0, 0, 0,
                                                   plane.fmt.bytesperpixel * plane.fmt.width, plane.fmt.height);
                    }

                    out_size = src->width * src->height * 3 / 2;
                    dst = abcdk_object_alloc2(out_size);
                    if (!dst)
                        return NULL;

                    chk = m_ctx->encodeFromBuffer(buffer, JCS_YCbCr, (uint8_t *)&dst->pptrs[0], out_size, m_quality);
                    if (chk != 0)
                    {
                        abcdk_object_unref(&dst);
                        return NULL;
                    }

                    /*真实长度。*/
                    dst->sizes[0] = out_size;

                    return dst;
                }

                virtual int update(const char *dst, const abcdk_torch_image_t *src)
                {
                    abcdk_object_t *dst_data;
                    ssize_t save_chk;

                    assert(dst != NULL && src != NULL);

                    dst_data = update(src);
                    if (!dst_data)
                        return -1;

                    truncate(dst, 0);

                    save_chk = abcdk_save(dst, dst_data->pptrs[0], dst_data->sizes[0], 0);
                    if (save_chk != dst_data->sizes[0])
                    {
                        abcdk_object_unref(&dst_data);
                        return -1;
                    }

                    abcdk_object_unref(&dst_data);
                    return 0;
                }
            };
        } // namespace jpeg
    } // namespace torch_cuda
} // namespace abcdk

#endif //__aarch64__
#endif // __cuda_cuda_h__

#endif // ABCDK_TORCH_NVIDIA_JPEG_ENCODER_AARCH64_HXX