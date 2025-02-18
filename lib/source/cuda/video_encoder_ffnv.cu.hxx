/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_VIDEO_ENCODER_FFNV_HXX
#define ABCDK_CUDA_VIDEO_ENCODER_FFNV_HXX

#include "abcdk/util/option.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/avutil.h"
#include "video_encoder.cu.hxx"
#include "video_util.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H
#ifdef AVCODEC_AVCODEC_H
#ifdef FFNV_CUDA_DYNLINK_LOADER_H

namespace abcdk
{
    namespace cuda
    {
        namespace video
        {
            class encoder_ffnv : public encoder
            {
            public:
                static encoder *create()
                {
                    encoder *ctx = new encoder_ffnv();
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

                    delete (encoder_ffnv *)ctx_p;
                }

            public:
                encoder_ffnv()
                {
                }

                virtual ~encoder_ffnv()
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

#endif // FFNV_CUDA_DYNLINK_LOADER_H
#endif // AVCODEC_AVCODEC_H
#endif // AVUTIL_AVUTIL_H
#endif // __cuda_cuda_h__

#endif // ABCDK_CUDA_VIDEO_ENCODER_FFNV_HXX