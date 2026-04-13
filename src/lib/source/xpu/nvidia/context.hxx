/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_NVIDIA_COMTEXT_HXX
#define ABCDK_XPU_NVIDIA_COMTEXT_HXX

#include "abcdk/util/trace.h"
#include "abcdk/util/atomic.h"
#include "abcdk/xpu/context.h"
#include "../base.in.h"

namespace abcdk_xpu
{
    namespace nvidia
    {
        namespace context
        {
            typedef struct _metadata
            {
                int device_id;
                CUcontext cu_ctx;
                int refcount;
            }metadata_t;
        
            static inline void unref(metadata_t **ctx)
            {
                metadata_t *ctx_p;

                if(!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;
                
                if(abcdk_atomic_add_and_fetch(&ctx_p->refcount,-1) != 0)
                    return;

                cuCtxDestroy(ctx_p->cu_ctx);

                delete ctx_p;
            }

            static inline metadata_t *refer(metadata_t *ctx)
            {
                abcdk_atomic_fetch_and_add(&ctx->refcount,1);
                return ctx;
            }

            static inline metadata_t *alloc(int id)
            {
                metadata_t *ctx;
                CUdevice cu_dev;
                CUcontext cu_ctx;
                CUresult cu_chk;

                cu_chk = cuDeviceGet(&cu_dev, id);
                if (cu_chk != CUDA_SUCCESS)
                    return NULL;
                    
#if CUDA_VERSION >= 13000
                cu_chk = cuCtxCreate(&cu_ctx, NULL, 0, cu_dev);
#else //#if CUDA_VERSION >= 13000
                cu_chk = cuCtxCreate(&cu_ctx, 0, cu_dev);
#endif //#if CUDA_VERSION >= 13000
                if (cu_chk != CUDA_SUCCESS)
                    return NULL;
                
                ctx = new metadata_t;
                if(!ctx)
                {
                     cuCtxDestroy(cu_ctx);
                    return NULL;
                }

                ctx->device_id = id;
                ctx->cu_ctx = cu_ctx;
                ctx->refcount = 1;

                return ctx;
            }

            static inline int current_push(metadata_t *ctx)
            {
                CUresult cu_chk;

                cu_chk = cuCtxPushCurrent(ctx->cu_ctx);
                if (cu_chk != CUDA_SUCCESS)
                    return -1;

                return 0;
            }

            static inline int current_pop(metadata_t *ctx)
            {
                std::queue<CUcontext> cu_ctx_stack;
                CUcontext cu_ctx = NULL;
                CUresult cu_chk;

                while(1)
                {
                    cu_chk = cuCtxPopCurrent(&cu_ctx);
                    if (cu_chk != CUDA_SUCCESS)
                        return -1;

                     /*出栈项为空或是自己时, 直接跳出.*/
                    if (ctx->cu_ctx == NULL || ctx->cu_ctx == cu_ctx)
                        break;
                    
                    /*出栈项栈是其它时, 入临时栈.*/
                    cu_ctx_stack.push(cu_ctx);
                }

                /*恢复栈信息.*/
                while (cu_ctx_stack.size() > 0)
                {
                    cuCtxPushCurrent(cu_ctx_stack.front());
                    cu_ctx_stack.pop();
                }

                ABCDK_TRACE_ASSERT(ctx->cu_ctx == cu_ctx,ABCDK_GETTEXT("设备环境的绑定与解绑必须配对使用."));

                return 0;
            }
        } // namespace context
    } // namespace nvidia
} // namespace abcdk_xpu

#endif //ABCDK_XPU_NVIDIA_COMTEXT_HXX