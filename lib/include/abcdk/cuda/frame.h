/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_FRAME_H
#define ABCDK_CUDA_FRAME_H

#include "abcdk/media/frame.h"
#include "abcdk/cuda/image.h"

__BEGIN_DECLS


/**
 * 重置。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_cuda_frame_reset(abcdk_media_frame_t **ctx, int width, int height, int pixfmt, int align);

/**创建。*/
abcdk_media_frame_t *abcdk_cuda_frame_create(int width, int height, int pixfmt, int align);

__END_DECLS

#endif //ABCDK_CUDA_FRAME_H
