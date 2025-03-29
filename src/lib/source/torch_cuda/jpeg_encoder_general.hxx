/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_JPEG_ENCODER_GENERAL_HXX
#define ABCDK_TORCH_NVIDIA_JPEG_ENCODER_GENERAL_HXX

#include "abcdk/util/option.h"
#include "abcdk/torch/image.h"
#include "abcdk/torch/context.h"
#include "abcdk/torch/nvidia.h"
#include "jpeg_encoder.hxx"

#ifdef __cuda_cuda_h__
#ifdef __x86_64__

namespace abcdk
{
    namespace torch_cuda
    {
        namespace jpeg
        {
            class encoder_general : public encoder
            {
            public:
                static encoder *create(CUcontext cuda_ctx)
                {
                    encoder *ctx = new encoder_general(cuda_ctx);
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

                    delete (encoder_general *)ctx_p;
                }

            private:
                CUcontext m_gpu_ctx;

                cudaStream_t m_stream;
                nvjpegHandle_t m_ctx;
                nvjpegEncoderState_t m_state;
                nvjpegEncoderParams_t m_params;

            public:
                encoder_general(CUcontext cuda_ctx)
                {
                    m_gpu_ctx = cuda_ctx;

                    m_stream = NULL;
                    m_ctx = NULL;
                    m_state = NULL;
                    m_params = NULL;
                }

                virtual ~encoder_general()
                {
                    close();
                }

            public:
                virtual void close()
                {
                    cuCtxPushCurrent(m_gpu_ctx);

                    if (m_params)
                        nvjpegEncoderParamsDestroy(m_params);
                    m_params = NULL;
                    if (m_state)
                        nvjpegEncoderStateDestroy(m_state);
                    m_state = NULL;
                    if (m_ctx)
                        nvjpegDestroy(m_ctx);
                    m_ctx = NULL;
                    if (m_stream)
                        cudaStreamDestroy(m_stream);
                    m_stream = NULL;

                    cuCtxPopCurrent(NULL);
                }

                virtual int open(abcdk_torch_jcodec_param_t *param)
                {
                    int quality;
                    cudaError_t cuda_chk;
                    nvjpegStatus_t jpeg_chk;

                    assert(param != NULL);

                    quality = ABCDK_CLAMP(param->quality, 1, 99);

                    abcdk::torch_cuda::context_robot robot(m_gpu_ctx);

                    cuda_chk = cudaStreamCreateWithFlags(&m_stream, cudaStreamDefault);
                    if (cuda_chk != cudaSuccess)
                        return -1;

                    jpeg_chk = nvjpegCreateSimple(&m_ctx);
                    if (jpeg_chk != NVJPEG_STATUS_SUCCESS)
                        return -1;

                    nvjpegEncoderStateCreate(m_ctx, &m_state, m_stream);
                    nvjpegEncoderParamsCreate(m_ctx, &m_params, m_stream);
                    nvjpegEncoderParamsSetQuality(m_params, quality, m_stream);
                    nvjpegEncoderParamsSetSamplingFactors(m_params, NVJPEG_CSS_420, m_stream);

                    return 0;
                }

                virtual abcdk_object_t *update(const abcdk_torch_image_t *src)
                {
                    abcdk_torch_image_t *tmp_src = NULL;
                    abcdk_object_t *dst;
                    size_t dst_size = 0;
                    nvjpegImage_t src_data = {0};
                    nvjpegStatus_t jpeg_chk;
                    int chk;

                    assert(src != NULL);

                    abcdk::torch_cuda::context_robot robot(m_gpu_ctx);

                    if (src->pixfmt != ABCDK_TORCH_PIXFMT_RGB24 && src->pixfmt != ABCDK_TORCH_PIXFMT_BGR24)
                    {
                        tmp_src = abcdk_torch_image_create_cuda(src->width, src->height, ABCDK_TORCH_PIXFMT_RGB24, 1);
                        if (!tmp_src)
                            return NULL;

                        chk = abcdk_torch_image_convert_cuda(tmp_src, src); // 转换格式。

                        if (chk == 0)
                            dst = update(tmp_src);

                        abcdk_torch_image_free_cuda(&tmp_src);

                        return dst;
                    }

                    src_data.channel[0] = src->data[0];
                    src_data.pitch[0] = src->stride[0];

                    if (src->pixfmt == ABCDK_TORCH_PIXFMT_RGB24)
                        jpeg_chk = nvjpegEncodeImage(m_ctx, m_state, m_params, &src_data, NVJPEG_INPUT_RGBI, src->width, src->height, m_stream);
                    else if (src->pixfmt == ABCDK_TORCH_PIXFMT_BGR24)
                        jpeg_chk = nvjpegEncodeImage(m_ctx, m_state, m_params, &src_data, NVJPEG_INPUT_BGRI, src->width, src->height, m_stream);

                    if (jpeg_chk != NVJPEG_STATUS_SUCCESS)
                        return NULL;

                    jpeg_chk = nvjpegEncodeRetrieveBitstream(m_ctx, m_state, NULL, &dst_size, m_stream);
                    if (jpeg_chk != NVJPEG_STATUS_SUCCESS)
                        return NULL;

                    dst = abcdk_object_alloc2(dst_size);
                    if (!dst)
                        return NULL;

                    jpeg_chk = nvjpegEncodeRetrieveBitstream(m_ctx, m_state, dst->pptrs[0], &dst_size, m_stream);
                    if (jpeg_chk != NVJPEG_STATUS_SUCCESS)
                        goto ERR;

                    return dst;

                ERR:

                    abcdk_object_unref(&dst);

                    return NULL;
                }

                virtual int update(const char *dst, const abcdk_torch_image_t *src)
                {
                    abcdk_object_t *dst_data;
                    ssize_t save_chk;
                    int chk;

                    assert(dst != NULL && src != NULL);

                    dst_data = update(src);
                    if (!dst_data)
                        return -1;

                    if (access(dst, F_OK) == 0)
                    {
                        chk = truncate(dst, 0);
                        if (chk != 0)
                            return -1;
                    }

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

#endif //__x86_64__
#endif // __cuda_cuda_h__

#endif // ABCDK_TORCH_NVIDIA_JPEG_ENCODER_GENERAL_HXX