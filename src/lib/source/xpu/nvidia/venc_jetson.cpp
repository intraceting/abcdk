/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"
#include "abcdk/util/trace.h"
#include "venc.hxx"

#ifdef __XPU_NVIDIA__MMAPI__

#include "NvVideoEncoder.h"

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace venc
        {
            typedef struct _metadata
            {

            } metadata_t;

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

            int get_params(metadata_t *ctx, abcdk_xpu_vcodec_params_t *params)
            {
                                
                return 0;
            }

            int recv_packet(metadata_t *ctx, abcdk_object_t **dst, int64_t *ts)
            {
                return -1;
            }

            int send_frame(metadata_t *ctx, const image::metadata_t *src, int64_t ts)
            {
                return -1;
            }
        } // namespace venc
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // #ifdef __XPU_NVIDIA__MMAPI__