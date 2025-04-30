/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/imgproc.h"
#include "abcdk/torch/opencv.h"
#include "abcdk/torch/dnnutil.h"

__BEGIN_DECLS

#ifdef OPENCV_DNN_HPP

int abcdk_torch_dnn_model_forward_host(const char *dst,const char *src, abcdk_option_t *opt)
{
    return -1;
}

#else // OPENCV_DNN_HPP

int abcdk_torch_dnn_model_forward_host(const char *dst,const char *src, abcdk_option_t *opt)
{
    abcdk_trace_printf(LOG_WARNING, TT("当前环境在构建时未包含OpenCV工具。"));
    return -1;
}

#endif // OPENCV_DNN_HPP

__END_DECLS