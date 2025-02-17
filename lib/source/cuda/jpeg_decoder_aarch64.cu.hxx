/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_JPEG_DECODER_AARCH64_HXX
#define ABCDK_CUDA_JPEG_DECODER_AARCH64_HXX

#include "abcdk/util/option.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/avutil.h"
#include "util.cu.hxx"
#include "jpeg_decoder.cu.hxx"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H
#ifdef __aarch64__

namespace abcdk
{
    namespace cuda
    {
        namespace jpeg
        {
            class decoder_aarch64:public decoder
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

                virtual AVFrame * update(const void *src, int src_size)
                {
                    return NULL;
                }

                virtual AVFrame * update(const void *src)
                {
                    return NULL;
                }
            };
        } // namespace jpeg
    } // namespace cuda
} // namespace abcdk

#endif //__aarch64__
#endif //AVUTIL_AVUTIL_H
#endif // __cuda_cuda_h__

#endif // ABCDK_CUDA_JPEG_DECODER_AARCH64_HXX