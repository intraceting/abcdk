/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_MEMORY_HXX
#define ABCDK_TORCH_MEMORY_HXX

#include "invoke.hxx"

namespace abcdk
{
    namespace torch
    {
        namespace memory
        {
            template <typename T>
            static inline void delete_object(T **ctx)
            {
                T *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                delete ctx_p;
            }

            template <typename T>
            static inline void delete_array(T **ctx)
            {
                T *ctx_p;

                if (!ctx || !*ctx)
                    return;

                ctx_p = *ctx;
                *ctx = NULL;

                delete[] ctx_p;
            }

        } // namespace memory

    } // namespace torch

} // namespace abcdk

#endif // ABCDK_TORCH_UTIL_HXX