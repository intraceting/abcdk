/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_JPEG_ENCODER_HXX
#define ABCDK_CUDA_JPEG_ENCODER_HXX

#include "abcdk/util/option.h"
#include "abcdk/media/jcodec.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/image.h"
#include "jpeg_util.cu.hxx"

#ifdef __cuda_cuda_h__

namespace abcdk
{
    namespace cuda
    {
        namespace jpeg
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

                virtual int open(abcdk_media_jcodec_param_t *param) = 0;

                virtual abcdk_object_t * update(const abcdk_media_image_t *src) = 0;

                virtual int update(const char *dst , const abcdk_media_image_t *src) = 0;
            };
        } // namespace jpeg
    } // namespace cuda
} // namespace abcdk


#endif // __cuda_cuda_h__

#endif // ABCDK_CUDA_JPEG_ENCODER_HXX