/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_VCODEC_DECODER_HXX
#define ABCDK_CUDA_VCODEC_DECODER_HXX

#include "abcdk/util/option.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/frame.h"
#include "abcdk/media/packet.h"
#include "abcdk/media/vcodec.h"
#include "vcodec_util.cu.hxx"

#ifdef __cuda_cuda_h__

namespace abcdk
{
    namespace cuda
    {
        namespace vcodec
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

                virtual int open(abcdk_media_vcodec_param_t *param) = 0;

                virtual int update(abcdk_media_frame_t **dst, const abcdk_media_packet_t *src) = 0;
            };
        } // namespace vcodec
    } // namespace cuda
} // namespace abcdk


#endif // __cuda_cuda_h__

#endif // ABCDK_CUDA_VCODEC_DECODER_HXX