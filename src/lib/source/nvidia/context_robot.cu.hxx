/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_NVIDIA_CONTEXT_ROBOT_HXX
#define ABCDK_NVIDIA_CONTEXT_ROBOT_HXX

#include "abcdk/util/trace.h"
#include "abcdk/nvidia/nvidia.h"

#ifdef __cuda_cuda_h__

namespace abcdk
{
    namespace cuda
    {
        namespace context
        {
            /*环境机器人，自动执行入栈和出栈。*/
            class robot
            {
            public:
                robot(CUcontext ctx)
                {
                    assert(ctx != NULL);
                    cuCtxPushCurrent(ctx);
                }

                virtual ~robot()
                {
                    cuCtxPopCurrent(NULL);
                }
            };
        } // namespace context
    } // namespace cuda
} // namespace abcdk

#endif //__cuda_cuda_h__

#endif // ABCDK_NVIDIA_CONTEXT_ROBOT_HXX