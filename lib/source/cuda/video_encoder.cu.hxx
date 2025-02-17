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

                virtual int open(enum AVCodecID codecid, void *ext_data, int ext_size) = 0;

                virtual int send(AVFrame *src) = 0;

                virtual AVPacket *recv() = 0;
            };
        } // namespace video
    } // namespace cuda
} // namespace abcdk

#endif // FFNV_CUDA_DYNLINK_LOADER_H
#endif // AVCODEC_AVCODEC_H
#endif // AVUTIL_AVUTIL_H
#endif // __cuda_cuda_h__

#endif // ABCDK_CUDA_VIDEO_ENCODER_HXX