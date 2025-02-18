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
            public:
                decoder_aarch64()
                {
                }

                virtual ~decoder_aarch64()
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

                virtual int update(AVFrame **dst, const AVPacket *src)
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

#endif // ABCDK_CUDA_VIDEO_DECODER_AARCH64_HXX