/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_NVIDIA_JPEG_ENCODER_HXX
#define ABCDK_NVIDIA_JPEG_ENCODER_HXX

#include "abcdk/util/option.h"
#include "abcdk/torch/jcodec.h"
#include "abcdk/nvidia/nvidia.h"
#include "abcdk/nvidia/image.h"
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

                virtual int open(abcdk_torch_jcodec_param_t *param) = 0;

                virtual abcdk_object_t * update(const abcdk_torch_image_t *src) = 0;

                virtual int update(const char *dst , const abcdk_torch_image_t *src) = 0;
            };
        } // namespace jpeg
    } // namespace cuda
} // namespace abcdk


#endif // __cuda_cuda_h__

#endif // ABCDK_NVIDIA_JPEG_ENCODER_HXX