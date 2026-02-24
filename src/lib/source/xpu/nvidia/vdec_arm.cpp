/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "vdec.hxx"

#ifdef __aarch64__

#include "jetson/nvmpi.h"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace vdec
        {
            typedef struct _metadata
            {
            }metadata_t;

            void free(metadata_t **ctx)
            {

            }

            metadata_t *alloc()
            {
                
            }

        } // namespace vdec
    } // namespace nvidia

} // namespace abcdk_xpu

#endif //#ifdef __aarch64__ 