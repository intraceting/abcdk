/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_FRAME_H
#define ABCDK_CUDA_FRAME_H

#include "abcdk/media/frame.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/memory.h"
#include "abcdk/cuda/imgproc.h"
#include "abcdk/cuda/image.h"

__BEGIN_DECLS

/**申请。*/
abcdk_media_frame_t *abcdk_cuda_frame_alloc();

__END_DECLS

#endif // ABCDK_CUDA_IMAGE_H