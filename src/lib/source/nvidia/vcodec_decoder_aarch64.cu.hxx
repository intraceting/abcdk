/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_NVIDIA_VIDEO_DECODER_AARCH64_HXX
#define ABCDK_NVIDIA_VIDEO_DECODER_AARCH64_HXX

#include "abcdk/torch/vcodec.h"
#include "abcdk/nvidia/nvidia.h"
#include "abcdk/nvidia/image.h"
#include "abcdk/nvidia/device.h"
#include "vcodec_decoder.cu.hxx"
#include "vcodec_util.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef __aarch64__

namespace abcdk
{
    namespace cuda
    {
        namespace vcodec
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

                static void frame_queue_destroy_cb(void *msg)
                {
                    av_frame_free((abcdk_torch_image_t **)&msg);
                }

            private:
                CUcontext m_gpu_ctx;

                nvmpictx *m_decoder;

            public:
                decoder_aarch64(CUcontext cuda_ctx)
                {
                    m_gpu_ctx = cuda_ctx;
                    m_decoder = NULL;

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

                    if (m_decoder)
                        nvmpi_decoder_close(m_decoder);
                    m_decoder = NULL;

                    if (m_gpu_ctx)
                        cuCtxPopCurrent(NULL);

                }

                virtual int open(abcdk_torch_vcodec_param_t *param)
                {
                    assert(param != NULL);

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    m_decoder = nvmpi_create_decoder((nvCodingType)vcodec_to_nvcodec(param->format), NV_PIX_YUV420);
                    if (!m_decoder)
                        return -1;

                    return 0;
                }

                virtual int update(abcdk_torch_frame_t **dst, const abcdk_torch_packet_t *src)
                {
                    nvPacket packet = {0};
                    nvFrame frame = {0};
                    bool get_wait = false;
                    int chk;

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    /*此平台不支持空的数据帧。*/
                    if (src && src->size > 0)
                    {
                        packet.payload_size = src->size;
                        packet.payload = src->data;
                        packet.pts = src->pts;
                        chk = nvmpi_decoder_put_packet(m_decoder, &packet);
                        if (chk != 0)
                            return -1;
                    }

                    if (dst)
                    {
                        /*src == NULL 表示结束包，要等待解码完成。*/
                        chk = nvmpi_decoder_get_frame(m_decoder, &frame, (src == NULL || src->size <= 0));
                        if (chk != 0)
                            return 0;

                        chk = abcdk_cuda_frame_reset(dst,frame.width, frame.height, AV_PIX_FMT_YUV420P, 1);
                        if (chk != 0)
                            return -1;

                        abcdk_cuda_imgutil_copy((*dst)->img->data, (*dst)->img->stride, 0, frame.payload, (int *)frame.linesize, 0, frame.width, frame.height, AV_PIX_FMT_YUV420P);

                        return 1;
                    }

                    return 0;
                }
            };
        } // namespace vcodec
    } // namespace cuda
} // namespace abcdk

#endif // __aarch64__
#endif // __cuda_cuda_h__

#endif // ABCDK_NVIDIA_VCODEC_DECODER_AARCH64_HXX