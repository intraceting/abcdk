/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_JPEG_DECODER_HXX
#define ABCDK_CUDA_JPEG_DECODER_HXX


#include "abcdk/util/option.h"
#include "abcdk/cuda/cuda.h"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

namespace abcdk
{
    namespace cuda
    {
        namespace jpeg
        {
            class decoder
            {
            protected:
                decoder()
                {

                }

                virtual ~decoder()
                {
                    close();
                }

            public:
                virtual void close()
                {
                    
                }

                virtual int open(abcdk_option_t *cfg) = 0;

                virtual AVFrame * update(const void *src, int src_size) = 0;

                virtual AVFrame * update(const void *src) = 0;
            };
        } // namespace jpeg
    } // namespace cuda
} // namespace abcdk

#endif //AVUTIL_AVUTIL_H
#endif // __cuda_cuda_h__

#endif // ABCDK_CUDA_JPEG_DECODER_HXX