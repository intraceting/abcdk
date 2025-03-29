/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_NVIDIA_CONTEXT_ROBOT_HXX
#define ABCDK_TORCH_NVIDIA_CONTEXT_ROBOT_HXX

#include "abcdk/util/trace.h"
#include "abcdk/torch/context.h"
#include "abcdk/torch/nvidia.h"

#ifdef __cuda_cuda_h__

namespace abcdk
{
    namespace torch_cuda
    {
        /*环境机器人，自动执行入栈和出栈。*/
        class context_robot
        {
        public:
            context_robot(CUcontext ctx)
            {
                cuCtxPushCurrent(ctx);
            }

            virtual ~context_robot()
            {
                cuCtxPopCurrent(NULL);
            }
        };

    } // namespace torch_cuda
} // namespace abcdk

#endif //__cuda_cuda_h__

#endif // ABCDK_TORCH_NVIDIA_CONTEXT_ROBOT_HXX