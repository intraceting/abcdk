/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_JPEG_DECODER_HXX
#define ABCDK_TORCH_NVIDIA_JPEG_DECODER_HXX

#include "abcdk/util/option.h"
#include "abcdk/torch/jcodec.h"
#include "abcdk/torch/nvidia.h"
#include "abcdk/torch/image.h"
#include "jpeg_util.cu.hxx"

#ifdef __cuda_cuda_h__

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

                virtual int open(abcdk_torch_jcodec_param_t *param) = 0;

                virtual abcdk_torch_image_t * update(const void *src, int src_size) = 0;

                virtual abcdk_torch_image_t * update(const void *src) = 0;
            };
        } // namespace jpeg
    } // namespace cuda
} // namespace abcdk


#endif // __cuda_cuda_h__

#endif // ABCDK_TORCH_NVIDIA_JPEG_DECODER_HXX