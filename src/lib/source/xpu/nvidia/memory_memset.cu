/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "memory.hxx"


namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace memory
        {
            template <typename T>
            __ABCDK_XPU_INVOKE_GLOBAL void _cudaMemset_3d3d(T *data, T value, size_t size)
            {
                size_t tid = common::util::kernel_thread_get_id();

                if (tid >= size)
                    return;

                data[tid] = value;
            }

            void *cudaMemset(void *dst, int val, size_t size)
            {
                dim3 grid, block;
                common::util::kernel_dim_make_3d3d(grid,block,size);

                _cudaMemset_3d3d<uint8_t><<<grid,block>>>((uint8_t *)dst, (uint8_t)val, size);

                return dst;
            }
        } // namespace memory
    } // namespace nvidia

} // namespace abcdk_xpu

