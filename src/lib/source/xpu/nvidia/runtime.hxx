/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_NVIDIA_RUNTIME_HXX
#define ABCDK_XPU_NVIDIA_RUNTIME_HXX

#include "abcdk/xpu/runtime.h"
#include "../runtime.in.h"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace runtime
        {
            static inline int deinit()
            {
                return 0;
            }

            static inline int init()
            {
                CUresult chk;

                chk = cuInit(0);
                if (chk != CUDA_SUCCESS)
                    return -1;

                return 0;
            }
        } // namespace runtime
    } // namespace nvidia

} // namespace abcdk_xpu

#endif //ABCDK_XPU_NVIDIA_RUNTIME_HXX