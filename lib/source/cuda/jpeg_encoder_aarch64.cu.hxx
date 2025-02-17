/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_JPEG_ENCODER_AARCH64_HXX
#define ABCDK_CUDA_JPEG_ENCODER_AARCH64_HXX

#include "abcdk/util/option.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/avutil.h"
#include "util.cu.hxx"
#include "jpeg_encoder.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H
#ifdef __aarch64__

namespace abcdk
{
    namespace cuda
    {
        namespace jpeg
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

                virtual abcdk_object_t *update(const AVFrame *src)
                {
                    return NULL;
                }

                virtual int update(const char *dst , const AVFrame *src)
                {
                    return -1;
                }
            };
        } // namespace jpeg
    } // namespace cuda
} // namespace abcdk

#endif //__aarch64__
#endif // AVUTIL_AVUTIL_H
#endif // __cuda_cuda_h__

#endif // ABCDK_CUDA_JPEG_ENCODER_AARCH64_HXX