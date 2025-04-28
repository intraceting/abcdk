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
            void delete_object(T **obj)
            {
                T *obj_p;

                if (!obj || !*obj)
                    return;

                obj_p = *obj;
                *obj = NULL;

                delete obj_p;
            }

            template <typename T>
            void delete_array(T **obj)
            {
                T *obj_p;

                if (!obj || !*obj)
                    return;

                obj_p = *obj;
                *obj = NULL;

                delete [] obj_p;
            }

        } // namespace memory 

    } // namespace torch

} // namespace abcdk

#endif // ABCDK_TORCH_UTIL_HXX