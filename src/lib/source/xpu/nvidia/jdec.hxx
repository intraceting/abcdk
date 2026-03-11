/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_NVIDIA_JDEC_HXX
#define ABCDK_XPU_NVIDIA_JDEC_HXX

#include "../base.in.h"
#include "../common/imgproc.hxx"
#include "image.hxx"
#include "imgproc.hxx"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace jdec
        {
            typedef struct _metadata metadata_t;

            void free(metadata_t **ctx);
            
            metadata_t *alloc();

            image::metadata_t *decode(metadata_t *ctx,const void *src, int src_size);
        } // namespace jdec
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // ABCDK_XPU_NVIDIA_JDEC_HXX