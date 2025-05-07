/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_INTER_MODE_HXX
#define ABCDK_TORCH_NVIDIA_INTER_MODE_HXX

#include "abcdk/torch/torch.h"
#include "abcdk/torch/nvidia.h"

#ifdef __cuda_cuda_h__

namespace abcdk
{
    namespace torch_cuda
    {
        namespace inter_mode
        {
            NppiInterpolationMode convert2nppi(int mode)
            {
                if (mode == ABCDK_TORCH_INTER_NEAREST)
                    return NPPI_INTER_NN;
                else if (mode == ABCDK_TORCH_INTER_LINEAR)
                    return NPPI_INTER_LINEAR;
                else if (mode == ABCDK_TORCH_INTER_CUBIC)
                    return NPPI_INTER_CUBIC;
                else
                    return NPPI_INTER_UNDEFINED;
            }

        } // namespace inter_mode

    } // namespace torch_cuda
} // namespace abcdk

#endif //__cuda_cuda_h__

#endif // ABCDK_TORCH_NVIDIA_INTER_MODE_HXX