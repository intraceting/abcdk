/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_GENERAL_DNN_INFER_HXX
#define ABCDK_XPU_GENERAL_DNN_INFER_HXX

#include "abcdk/xpu/image.h"
#include "../runtime.in.h"
#include "../common/util.hxx"
#include "dnn_engine.hxx"

namespace abcdk_xpu
{
    namespace general
    {
        namespace dnn
        {
            namespace infer
            {
                typedef struct _metadata metadata_t;

                void free(metadata_t **ctx);

                metadata_t *alloc();

                int load_model(metadata_t *ctx, const char *file, abcdk_option_t *opt);

                int fetch_tensor(metadata_t *ctx, int count, abcdk_xpu_dnn_tensor_t tensor[]);

                int forward(metadata_t *ctx, int count, image::metadata_t *img[]);

            } // namespace infer
        } // namespace dnn
    } // namespace general
} // namespace abcdk_xpu

#endif // ABCDK_XPU_GENERAL_DNN_INFER_HXX