/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_VIDEO_ENCODER_HXX
#define ABCDK_CUDA_VIDEO_ENCODER_HXX

#include "abcdk/util/option.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/avutil.h"
#include "video_util.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H
#ifdef AVCODEC_AVCODEC_H

namespace abcdk
{
    namespace cuda
    {
        namespace video
        {
            class encoder
            {
            protected:
                encoder()
                {
                }

                virtual ~encoder()
                {
                    close();
                }

            public:
                virtual void close()
                {
                }

                virtual int open(abcdk_option_t *cfg) = 0;

                virtual int sync(AVCodecContext *opt) = 0;

                virtual int update(AVPacket **dst, const AVFrame *src) = 0;
            };
        } // namespace video
    } // namespace cuda
} // namespace abcdk


#endif // AVCODEC_AVCODEC_H
#endif // AVUTIL_AVUTIL_H
#endif // __cuda_cuda_h__

#endif // ABCDK_CUDA_VIDEO_ENCODER_HXX