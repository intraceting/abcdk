/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "vdec.hxx"

#ifdef __aarch64__

#include "jetson/NvVideoDecoder.h"

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
                return NULL;
            }

            int setup(metadata_t *ctx, const abcdk_xpu_vcodec_params_t *params, context::metadata_t *rt_ctx)
            {
                return -1;
            }

            int send_packet(metadata_t *ctx, const void *src_data, size_t src_size, int64_t ts)
            {
                return -1;
            }

            int recv_frame(metadata_t *ctx, image::metadata_t **dst, int64_t *ts)
            {
                return -1;
            }

        } // namespace vdec
    } // namespace nvidia

} // namespace abcdk_xpu

#endif //#ifdef __aarch64__ 