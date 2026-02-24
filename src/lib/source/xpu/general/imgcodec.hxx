/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_GENERAL_IMGCODEC_HXX
#define ABCDK_XPU_GENERAL_IMGCODEC_HXX

#include "abcdk/xpu/imgcodec.h"
#include "../runtime.in.h"
#include "../common/imgcodec.hxx"
#include "image.hxx"
#include "imgproc.hxx"

namespace abcdk_xpu
{
    namespace general
    {
        namespace imgcodec
        {
            abcdk_object_t *encode(const image::metadata_t *src, const char *ext);

            image::metadata_t *decode(const void *src, size_t size);
        } // namespace imgcodec
    } // namespace general
} // namespace abcdk_xpu

#endif //ABCDK_XPU_GENERAL_IMGCODEC_HXX