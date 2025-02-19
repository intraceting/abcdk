/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_VIDEO_DECODER_AARCH64_HXX
#define ABCDK_CUDA_VIDEO_DECODER_AARCH64_HXX

#include "abcdk/util/option.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/avutil.h"
#include "abcdk/cuda/device.h"
#include "video_decoder.cu.hxx"
#include "video_util.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H
#ifdef AVCODEC_AVCODEC_H
#ifdef __aarch64__

namespace abcdk
{
    namespace cuda
    {
        namespace video
        {
            class decoder_aarch64 : public decoder
            {
            public:
                static decoder *create()
                {
                    decoder *ctx = new decoder_aarch64();
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
                    av_frame_free((AVFrame **)&msg);
                }

            private:
                CUcontext m_gpu_ctx;

                nvmpictx *m_decoder;

                abcdk_option_t *m_cfg;

            public:
                decoder_aarch64()
                {
                    m_gpu_ctx = NULL;
                    m_decoder = NULL;

                    m_cfg = NULL;
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

                    abcdk_cuda_ctx_destroy(&m_gpu_ctx);

                    abcdk_option_free(&m_cfg);
                }

                virtual int open(abcdk_option_t *cfg)
                {
                    int device;
                    CUresult chk;

                    assert(m_cfg == NULL);

                    m_cfg = abcdk_option_alloc("--");
                    if (!m_cfg)
                        return -1;

                    if (cfg)
                        abcdk_option_merge(m_cfg, cfg);

                    device = abcdk_option_get_int(m_cfg, "--device", 0, 0);

                    m_gpu_ctx = abcdk_cuda_ctx_create(device, 0);
                    if (!m_gpu_ctx)
                        return -1;

                    return 0;
                }

                virtual int sync(AVCodecContext *opt)
                {
                    assert(opt != NULL);

                    abcdk::cuda::context::robot robot(m_gpu_ctx);

                    m_decoder = nvmpi_create_decoder((nvCodingType)codecid_ffmpeg_to_nvcodec(opt->codec_id), NV_PIX_YUV420);
                    if (!m_decoder)
                        return -1;

                    return 0;
                }

                virtual int update(AVFrame **dst, const AVPacket *src)
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

                        *dst = abcdk_cuda_avframe_alloc(frame.width, frame.height, AV_PIX_FMT_YUV420P, 1);
                        if (!*dst)
                            return -1;

                        abcdk_cuda_avimage_copy((*dst)->data, (*dst)->linesize, 0, frame.payload, (int *)frame.linesize, 0, frame.width, frame.height, AV_PIX_FMT_YUV420P);

                        return 1;
                    }

                    return 0;
                }
            };
        } // namespace video
    } // namespace cuda
} // namespace abcdk

#endif // __aarch64__
#endif // AVCODEC_AVCODEC_H
#endif // AVUTIL_AVUTIL_H
#endif // __cuda_cuda_h__

#endif // ABCDK_CUDA_VIDEO_DECODER_AARCH64_HXX