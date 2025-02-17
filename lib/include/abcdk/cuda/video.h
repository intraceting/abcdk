/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_VIDEO_H
#define ABCDK_CUDA_VIDEO_H

#include "abcdk/util/option.h"
#include "abcdk/util/object.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/device.h"
#include "abcdk/cuda/avutil.h"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H
#ifdef AVCODEC_AVCODEC_H

__BEGIN_DECLS

/**VIDEO编/解码器。*/
typedef struct _abcdk_cuda_video abcdk_cuda_video_t;


__END_DECLS

#endif //AVCODEC_AVCODEC_H
#endif //AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__

#endif // ABCDK_CUDA_VIDEO_H