/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_NVIDIA_JENC_HXX
#define ABCDK_XPU_NVIDIA_JENC_HXX

#include "../runtime.in.h"
#include "../common/imgproc.hxx"
#include "image.hxx"
#include "imgproc.hxx"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace jenc
        {
            typedef struct _metadata metadata_t;

            void free(metadata_t **ctx);
            
            metadata_t *alloc();

            abcdk_object_t *encode(metadata_t *ctx, const image::metadata_t *src);
        } // namespace jenc
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // ABCDK_XPU_NVIDIA_JENC_HXX