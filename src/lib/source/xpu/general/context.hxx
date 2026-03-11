/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_GENERAL_COMTEXT_HXX
#define ABCDK_XPU_GENERAL_COMTEXT_HXX

#include "abcdk/util/atomic.h"
#include "abcdk/xpu/context.h"
#include "../base.in.h"

namespace abcdk_xpu
{
    namespace general
    {
        namespace context
        {
            typedef struct _metadata
            {
                int device_id;
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

                ctx = new metadata_t;
                if(!ctx)
                    return NULL;

                ctx->device_id = id;
                ctx->refcount = 1;

                return ctx;
            }

            static inline int current_push(metadata_t *ctx)
            {
                return 0;
            }

            static inline int current_pop(metadata_t *ctx)
            {
                return 0;
            }
        } // namespace context
    } // namespace general
} // namespace abcdk_xpu

#endif //ABCDK_XPU_GENERAL_COMTEXT_HXX