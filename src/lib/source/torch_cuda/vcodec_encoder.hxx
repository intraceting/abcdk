/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_VCODEC_ENCODER_HXX
#define ABCDK_TORCH_NVIDIA_VCODEC_ENCODER_HXX

#include "abcdk/torch/vcodec.h"
#include "abcdk/torch/image.h"
#include "abcdk/torch/nvidia.h"
#include "vcodec_util.hxx"

#ifdef __cuda_cuda_h__

namespace abcdk
{
    namespace torch_cuda
    {
        namespace vcodec
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

                virtual int open(abcdk_torch_vcodec_param_t *param) = 0;

                virtual int update(abcdk_torch_packet_t **dst, const abcdk_torch_frame_t *src) = 0;
            };
        } // namespace vcodec
    } // namespace torch_cuda
} // namespace abcdk


#endif // __cuda_cuda_h__

#endif // ABCDK_TORCH_NVIDIA_VCODEC_ENCODER_HXX