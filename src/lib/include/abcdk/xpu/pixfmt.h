/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_PIXFMT_H
#define ABCDK_XPU_PIXFMT_H

#include "abcdk/xpu/types.h"
#include "abcdk/ffmpeg/util.h"

__BEGIN_DECLS

/**获取像素位宽.*/
int abcdk_xpu_pixfmt_get_bit(abcdk_xpu_pixfmt_t pixfmt, int have_pad);

/**获取像素格式名字. */
const char *abcdk_xpu_pixfmt_get_name(abcdk_xpu_pixfmt_t pixfmt);

/**获取像素格式通道数. */
int abcdk_xpu_pixfmt_get_channel(abcdk_xpu_pixfmt_t pixfmt);


__END_DECLS

#endif // ABCDK_XPU_PIXFMT_H