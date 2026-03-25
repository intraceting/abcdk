/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_NVIDIA_VDEC_HXX
#define ABCDK_XPU_NVIDIA_VDEC_HXX

#include "../base.in.h"
#include "../common/imgproc.hxx"
#include "../common/util.hxx"
#include "context.hxx"
#include "image.hxx"
#include "imgproc.hxx"
#include "vcodec_util.hxx"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace vdec
        {
            typedef struct _metadata metadata_t;

            void free(metadata_t **ctx);

            metadata_t *alloc();

            int setup(metadata_t *ctx, const abcdk_xpu_vcodec_params_t *params, context::metadata_t *rt_ctx);

            int send_packet(metadata_t *ctx, const void *src_data, size_t src_size, int64_t ts);

            int recv_frame(metadata_t *ctx, image::metadata_t **dst, int64_t *ts);
        } // namespace vdec
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // ABCDK_XPU_NVIDIA_VDEC_HXX