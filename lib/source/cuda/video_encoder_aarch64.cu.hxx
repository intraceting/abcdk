/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_VIDEO_ENCODER_AARCH64_HXX
#define ABCDK_CUDA_VIDEO_ENCODER_AARCH64_HXX

#include "abcdk/util/option.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/avutil.h"
#include "video_encoder.cu.hxx"
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
            class encoder_aarch64 : public encoder
            {
            public:
                static encoder *create()
                {
                    encoder *ctx = new encoder_aarch64();
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
            public:
                encoder_aarch64()
                {
                }

                virtual ~encoder_aarch64()
                {
                    close();
                }

            public:
                virtual void close()
                {
                }

                virtual int open(abcdk_option_t *cfg)
                {
                    return -1;
                }

                virtual int sync(AVCodecContext *opt)
                {
                    return -1;
                }

                virtual int update(AVPacket **dst, const AVFrame *src)
                {
                    return -1;
                }
            };
        } // namespace video
    } // namespace cuda
} // namespace abcdk

#endif // __aarch64__
#endif // AVCODEC_AVCODEC_H
#endif // AVUTIL_AVUTIL_H
#endif // __cuda_cuda_h__

#endif // ABCDK_CUDA_VIDEO_ENCODER_AARCH64_HXX