/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_GENERAL_STITCHER_HXX
#define ABCDK_XPU_GENERAL_STITCHER_HXX

#include "abcdk/xpu/stitcher.h"
#include "../base.in.h"
#include "../common/stitcher.hxx"
#include "../common/util.hxx"
#include "image.hxx"
#include "imgproc.hxx"

namespace abcdk_xpu
{
    namespace general
    {
        namespace stitcher
        {
            typedef struct _metadata metadata_t;

            void free(metadata_t **ctx);

            metadata_t *alloc();

            int set_feature_finder(metadata_t *ctx, const char *name);

            int set_warper(metadata_t *ctx, const char *name);

            int estimate_parameters(metadata_t *ctx, int count, const image::metadata_t *img[], const image::metadata_t *mask[], float threshold);

            int build_parameters(metadata_t *ctx);

            abcdk_object_t *dump_parameters(metadata_t *ctx, const char *magic);

            int load_parameters(metadata_t *ctx, const char *src,  const char *magic);

            int compose(metadata_t *ctx, int count, const image::metadata_t *img[], image::metadata_t **out ,int optimize_seam);
  
        }//namespace stitcher
    } // namespace general

} // namespace abcdk_xpu

#endif //ABCDK_XPU_GENERAL_STITCHER_HXX