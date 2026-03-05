/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_GENERAL_CALIBRATE_HXX
#define ABCDK_XPU_GENERAL_CALIBRATE_HXX

#include "abcdk/xpu/calibrate.h"
#include "../runtime.in.h"
#include "../common/calibrate.hxx"
#include "../common/util.hxx"
#include "image.hxx"
#include "imgproc.hxx"

namespace abcdk_xpu
{
    namespace general
    {
        namespace calibrate
        {
            typedef struct _metadata metadata_t;

            void free(metadata_t **ctx);

            metadata_t *alloc();
  
            void setup(metadata_t *ctx ,int board_cols,int board_rows,float grid_width,float grid_height);

            int detect_corners(metadata_t *ctx ,const image::metadata_t *img, int win_width, int win_height);

            double estimate_parameters(metadata_t *ctx);

            int build_parameters(metadata_t *ctx, double alpha = 1);

            abcdk_object_t *dump_parameters(metadata_t *ctx, const char *magic);

            int load_parameters(metadata_t *ctx, const char *src,  const char *magic);

            int undistort(metadata_t *ctx, const image::metadata_t *src, image::metadata_t **dst, abcdk_xpu_inter_t inter_mode);
        }//namespace calibrate
    } // namespace general
} // namespace abcdk_xpu

#endif //ABCDK_XPU_GENERAL_CALIBRATE_HXX