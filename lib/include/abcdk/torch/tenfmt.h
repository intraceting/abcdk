/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_TENFMT_H
#define ABCDK_TORCH_TENFMT_H

#include "abcdk/torch/torch.h"

__BEGIN_DECLS

/** 张量格式。*/
typedef enum _abcdk_torch_tenfmt_constant
{
    /** NCHW (AAA...BBB...CCC)*/
    ABCDK_TORCH_TENFMT_NCHW = 1,
#define ABCDK_TORCH_TENFMT_NCHW ABCDK_TORCH_TENFMT_NCHW

    /** NHWC (ABC...ABC...ABC)*/
    ABCDK_TORCH_TENFMT_NHWC = 2,
#define ABCDK_TORCH_TENFMT_NHWC ABCDK_TORCH_TENFMT_NHWC

}abcdk_torch_tenfmt_constant_t;

__END_DECLS

#endif // ABCDK_TORCH_TENFMT_H