/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_GENERAL_VENC_HXX
#define ABCDK_XPU_GENERAL_VENC_HXX

#include "abcdk/xpu/venc.h"
#include "../base.in.h"
#include "vcodec_util.hxx"
#include "context.hxx"
#include "image.hxx"
#include "imgproc.hxx"

namespace abcdk_xpu
{
    namespace general
    {
        namespace venc
        {
            typedef struct _metadata metadata_t;

            void free(metadata_t **ctx);

            metadata_t *alloc();

            int setup(metadata_t *ctx, const abcdk_xpu_vcodec_params_t *params, context::metadata_t *rt_ctx);

            int get_params(metadata_t *ctx, abcdk_xpu_vcodec_params_t *params);

            int recv_packet(metadata_t *ctx ,abcdk_object_t **dst, int64_t *ts);

            int send_frame(metadata_t *ctx ,const image::metadata_t *src, int64_t ts);
        } // namespace venc
    } // namespace general
} // namespace abcdk_xpu

#endif //ABCDK_XPU_GENERAL_VENC_HXX