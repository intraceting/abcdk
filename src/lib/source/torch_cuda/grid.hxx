/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_GRID_HXX
#define ABCDK_TORCH_NVIDIA_GRID_HXX

#include "abcdk/torch/torch.h"
#include "abcdk/torch/nvidia.h"
#include "../torch/invoke.hxx"

#ifdef __cuda_cuda_h__

namespace abcdk
{
    namespace torch_cuda
    {
        namespace grid
        {
            ABCDK_TORCH_INVOKE_HOST void make_dim(uint3 *dim, size_t n, size_t b)
            {
                size_t k;
                unsigned int x, y;

                assert(dim != NULL && n > 0 && b > 0);

                k = (n - 1) / b + 1;
                x = k;
                y = 1;

                if (x > 65535)
                {
                    x = ceil(sqrt(k));
                    y = (n - 1) / (x * b) + 1;
                }

                dim->x = x;
                dim->y = y;
                dim->z = 1;
            }

            ABCDK_TORCH_INVOKE_HOST void make_dim_dim(uint3 dim[2], size_t n, size_t b)
            {
                make_dim(&dim[1], b, 1);
                make_dim(&dim[0], n, dim[1].x * dim[1].y * dim[1].z);
            }

            ABCDK_TORCH_INVOKE_DEVICE size_t get_tid(size_t thread, size_t block)
            {
                /*thread 1D.*/
                if (thread == 1 && block == 0)
                    return threadIdx.x;

                /*thread 2D.*/
                if (thread == 2 && block == 0)
                    return threadIdx.x + threadIdx.y * blockDim.x;

                /*thread 3D.*/
                if (thread == 3 && block == 0)
                    return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

                /*block 1D.*/
                if (thread == 0 && block == 1)
                    return blockIdx.x;

                /*block 2D.*/
                if (thread == 0 && block == 2)
                    return blockIdx.x + blockIdx.y * gridDim.x;

                /*block 3D.*/
                if (thread == 0 && block == 3)
                    return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;

                /*block-thread 1D-1D.*/
                if (thread == 1 && block == 1)
                    return threadIdx.x + blockDim.x * blockIdx.x;

                /*block-thread 1D-2D.*/
                if (thread == 2 && block == 1)
                {
                    size_t threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
                    return threadId_2D + (blockDim.x * blockDim.y) * blockIdx.x;
                }

                /*block-thread 1D-3D.*/
                if (thread == 3 && block == 1)
                {
                    size_t threadId_3D = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
                    return threadId_3D + (blockDim.x * blockDim.y * blockDim.z) * blockIdx.x;
                }

                /*block-thread 2D-1D.*/
                if (thread == 1 && block == 2)
                {
                    size_t blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
                    return threadIdx.x + blockDim.x * blockId_2D;
                }

                /*block-thread 3D-1D.*/
                if (thread == 1 && block == 3)
                {
                    size_t blockId_3D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
                    return threadIdx.x + blockDim.x * blockId_3D;
                }

                /*block-thread 2D-2D.*/
                if (thread == 2 && block == 2)
                {
                    size_t threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
                    size_t blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
                    return threadId_2D + (blockDim.x * blockDim.y) * blockId_2D;
                }

                /*block-thread 2D-3D.*/
                if (thread == 3 && block == 2)
                {
                    size_t threadId_3D = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
                    size_t blockId_2D = blockIdx.x + blockIdx.y * gridDim.x;
                    return threadId_3D + (blockDim.x * blockDim.y * blockDim.z) * blockId_2D;
                }

                /*block-thread 3D-2D.*/
                if (thread == 2 && block == 3)
                {
                    size_t threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
                    size_t blockId_3D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
                    return threadId_2D + (blockDim.x * blockDim.y) * blockId_3D;
                }

                /*block-thread 3D-3D.*/
                if (thread == 3 && block == 3)
                {
                    size_t threadId_3D = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
                    size_t blockId_3D = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
                    return threadId_3D + (blockDim.x * blockDim.y * blockDim.z) * blockId_3D;
                }

                return (size_t)-1;
            }
        } // namespace grid

    } // namespace torch_cuda
} // namespace abcdk

#endif //__cuda_cuda_h__

#endif // ABCDK_TORCH_NVIDIA_GRID_HXX