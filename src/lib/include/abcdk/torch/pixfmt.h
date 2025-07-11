/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_PIXFMT_H
#define ABCDK_TORCH_PIXFMT_H

#include "abcdk/torch/torch.h"

__BEGIN_DECLS

/**常量。*/
typedef enum _abcdk_torch_pixfmt_constant
{
    ABCDK_TORCH_PIXFMT_NONE = -1,
#define ABCDK_TORCH_PIXFMT_NONE ABCDK_TORCH_PIXFMT_NONE

    ABCDK_TORCH_PIXFMT_YUV420P = 1, ///< planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples)
#define ABCDK_TORCH_PIXFMT_YUV420P ABCDK_TORCH_PIXFMT_YUV420P

    ABCDK_TORCH_PIXFMT_YUV420P9,
#define ABCDK_TORCH_PIXFMT_YUV420P9 ABCDK_TORCH_PIXFMT_YUV420P9

    ABCDK_TORCH_PIXFMT_YUV420P10,
#define ABCDK_TORCH_PIXFMT_YUV420P10 ABCDK_TORCH_PIXFMT_YUV420P10

    ABCDK_TORCH_PIXFMT_YUV420P12,
#define ABCDK_TORCH_PIXFMT_YUV420P12 ABCDK_TORCH_PIXFMT_YUV420P12

    ABCDK_TORCH_PIXFMT_YUV420P14,
#define ABCDK_TORCH_PIXFMT_YUV420P14 ABCDK_TORCH_PIXFMT_YUV420P14

    ABCDK_TORCH_PIXFMT_YUV420P16,
#define ABCDK_TORCH_PIXFMT_YUV420P16 ABCDK_TORCH_PIXFMT_YUV420P16

    ABCDK_TORCH_PIXFMT_YUV422P = 20, ///< planar YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
#define ABCDK_TORCH_PIXFMT_YUV422P ABCDK_TORCH_PIXFMT_YUV422P

    ABCDK_TORCH_PIXFMT_YUV422P9,
#define ABCDK_TORCH_PIXFMT_YUV422P9 ABCDK_TORCH_PIXFMT_YUV422P9

    ABCDK_TORCH_PIXFMT_YUV422P10,
#define ABCDK_TORCH_PIXFMT_YUV422P10 ABCDK_TORCH_PIXFMT_YUV422P10

    ABCDK_TORCH_PIXFMT_YUV422P12,
#define ABCDK_TORCH_PIXFMT_YUV422P12 ABCDK_TORCH_PIXFMT_YUV422P12

    ABCDK_TORCH_PIXFMT_YUV422P14,
#define ABCDK_TORCH_PIXFMT_YUV422P14 ABCDK_TORCH_PIXFMT_YUV422P14

    ABCDK_TORCH_PIXFMT_YUV422P16,
#define ABCDK_TORCH_PIXFMT_YUV422P16 ABCDK_TORCH_PIXFMT_YUV422P16

    ABCDK_TORCH_PIXFMT_YUV444P = 30, ///< planar YUV 4:4:4, 24bpp, (1 Cr & Cb sample per 1x1 Y samples)
#define ABCDK_TORCH_PIXFMT_YUV444P ABCDK_TORCH_PIXFMT_YUV444P

    ABCDK_TORCH_PIXFMT_YUV444P9,
#define ABCDK_TORCH_PIXFMT_YUV444P9 ABCDK_TORCH_PIXFMT_YUV444P9

    ABCDK_TORCH_PIXFMT_YUV444P10,
#define ABCDK_TORCH_PIXFMT_YUV444P10 ABCDK_TORCH_PIXFMT_YUV444P10

    ABCDK_TORCH_PIXFMT_YUV444P12,
#define ABCDK_TORCH_PIXFMT_YUV444P12 ABCDK_TORCH_PIXFMT_YUV444P12

    ABCDK_TORCH_PIXFMT_YUV444P14,
#define ABCDK_TORCH_PIXFMT_YUV444P14 ABCDK_TORCH_PIXFMT_YUV444P14

    ABCDK_TORCH_PIXFMT_YUV444P16,
#define ABCDK_TORCH_PIXFMT_YUV444P16 ABCDK_TORCH_PIXFMT_YUV444P16

    ABCDK_TORCH_PIXFMT_NV12 = 40, ///< planar YUV 4:2:0, 12bpp, 1 plane for Y and 1 plane for the UV components, which are interleaved (first byte U and the following byte V)
#define ABCDK_TORCH_PIXFMT_NV12 ABCDK_TORCH_PIXFMT_NV12

    ABCDK_TORCH_PIXFMT_P016, ///< like NV12, with 16bpp per component
#define ABCDK_TORCH_PIXFMT_P016 ABCDK_TORCH_PIXFMT_P016

    ABCDK_TORCH_PIXFMT_NV16, ///< interleaved chroma YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
#define ABCDK_TORCH_PIXFMT_NV16 ABCDK_TORCH_PIXFMT_NV16

    ABCDK_TORCH_PIXFMT_NV21, ///< as above, but U and V bytes are swapped
#define ABCDK_TORCH_PIXFMT_NV21 ABCDK_TORCH_PIXFMT_NV21

    ABCDK_TORCH_PIXFMT_NV24, ///< planar YUV 4:4:4, 24bpp, 1 plane for Y and 1 plane for the UV components, which are interleaved (first byte U and the following byte V)
#define ABCDK_TORCH_PIXFMT_NV24 ABCDK_TORCH_PIXFMT_NV24

    ABCDK_TORCH_PIXFMT_NV42, ///< as above, but U and V bytes are swapped
#define ABCDK_TORCH_PIXFMT_NV42 ABCDK_TORCH_PIXFMT_NV42

    ABCDK_TORCH_PIXFMT_GRAY8 = 50,
#define ABCDK_TORCH_PIXFMT_GRAY8 ABCDK_TORCH_PIXFMT_GRAY8

    ABCDK_TORCH_PIXFMT_GRAY16,
#define ABCDK_TORCH_PIXFMT_GRAY16 ABCDK_TORCH_PIXFMT_GRAY16

    ABCDK_TORCH_PIXFMT_GRAYF32,
#define ABCDK_TORCH_PIXFMT_GRAYF32 ABCDK_TORCH_PIXFMT_GRAYF32

    ABCDK_TORCH_PIXFMT_RGB24 = 60, ///< packed RGB 8:8:8, 24bpp, RGBRGB...
#define ABCDK_TORCH_PIXFMT_RGB24 ABCDK_TORCH_PIXFMT_RGB24

    ABCDK_TORCH_PIXFMT_BGR24, ///< packed RGB 8:8:8, 24bpp, BGRBGR...
#define ABCDK_TORCH_PIXFMT_BGR24 ABCDK_TORCH_PIXFMT_BGR24

    ABCDK_TORCH_PIXFMT_RGB32, 
#define ABCDK_TORCH_PIXFMT_RGB32 ABCDK_TORCH_PIXFMT_RGB32

    ABCDK_TORCH_PIXFMT_BGR32, 
#define ABCDK_TORCH_PIXFMT_BGR32 ABCDK_TORCH_PIXFMT_BGR32

} abcdk_torch_pixfmt_constant_t;

/**转为ffmpeg类型。*/
int abcdk_torch_pixfmt_convert_to_ffmpeg(int format);

/**从ffmpeg类型转。*/
int abcdk_torch_pixfmt_convert_from_ffmpeg(int format);

/**
 * 获取像素格式通道数。
 */
int abcdk_torch_pixfmt_channels(int format);


__END_DECLS

#endif // ABCDK_TORCH_PIXFMT_H