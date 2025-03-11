/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_NVIDIA_VCODEC_ENCODER_HXX
#define ABCDK_NVIDIA_VCODEC_ENCODER_HXX

#include "abcdk/torch/vcodec.h"
#include "abcdk/nvidia/nvidia.h"
#include "abcdk/nvidia/image.h"
#include "vcodec_util.cu.hxx"

#ifdef __cuda_cuda_h__

namespace abcdk
{
    namespace cuda
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
    } // namespace cuda
} // namespace abcdk


#endif // __cuda_cuda_h__

#endif // ABCDK_NVIDIA_VCODEC_ENCODER_HXX