/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/opencv.h"
#include "abcdk/torch/dnnengine.h"

__BEGIN_DECLS

#ifdef OPENCV_DNN_HPP

void abcdk_torch_dnn_engine_free_host(abcdk_torch_dnn_engine_t **ctx)
{

}

abcdk_torch_dnn_engine_t *abcdk_torch_dnn_engine_alloc_host()
{
    return NULL;
}

int abcdk_torch_dnn_engine_load_model_host(abcdk_torch_dnn_engine_t *ctx, const char *file, abcdk_option_t *opt)
{
    return -1;
}

int abcdk_torch_dnn_engine_fetch_tensor_host(abcdk_torch_dnn_engine_t *ctx, int count, abcdk_torch_dnn_tensor tensor[])
{
    return -1;
}

int abcdk_torch_dnn_engine_infer_host(abcdk_torch_dnn_engine_t *ctx, int count, abcdk_torch_image_t *img[])
{
    return -1;
}

#else // OPENCV_DNN_HPP

void abcdk_torch_dnn_engine_free_host(abcdk_torch_dnn_engine_t **ctx)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return ;
}

abcdk_torch_dnn_engine_t *abcdk_torch_dnn_engine_alloc_host()
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return NULL;
}

int abcdk_torch_dnn_engine_load_model_host(abcdk_torch_dnn_engine_t *ctx, const char *file, abcdk_option_t *opt)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1; 
}

int abcdk_torch_dnn_engine_fetch_tensor_host(abcdk_torch_dnn_engine_t *ctx, int count, abcdk_torch_dnn_tensor info[])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

int abcdk_torch_dnn_engine_infer_host(abcdk_torch_dnn_engine_t *ctx, int count, abcdk_torch_image_t *img[])
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1; 
}

#endif // OPENCV_DNN_HPP

__END_DECLS