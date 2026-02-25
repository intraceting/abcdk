/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "jenc.hxx"

#ifdef __aarch64__

namespace abcdk_xpu
{

    namespace nvidia
    {
        namespace jdec
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

            image::metadata_t *decode(metadata_t *ctx, const void *src, int src_size)
            {
                return NULL;
            }
        } // namespace jdec
    } // namespace nvidia

} // namespace abcdk_xpu

#endif // #ifdef __aarch64__