/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_FRAME_H
#define ABCDK_TORCH_FRAME_H

#include "abcdk/torch/image.h"

__BEGIN_DECLS

/**媒体帧。*/
typedef struct _abcdk_torch_frame
{
    /**数据(图像)。*/
    abcdk_torch_image_t *img;

    /**DTS.*/
    int64_t dts;

    /**PTS.*/
    int64_t pts;

}abcdk_torch_frame_t;

/**释放。 */
void abcdk_torch_frame_free(abcdk_torch_frame_t **ctx);

/**申请。*/
abcdk_torch_frame_t *abcdk_torch_frame_alloc();

/**
 * 重置。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_torch_frame_reset_host(abcdk_torch_frame_t **ctx, int width, int height, int pixfmt, int align);

/**
 * 重置。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_torch_frame_reset_cuda(abcdk_torch_frame_t **ctx, int width, int height, int pixfmt, int align);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_frame_reset abcdk_torch_frame_reset_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_frame_reset abcdk_torch_frame_reset_host
#endif //

/**创建。*/
abcdk_torch_frame_t *abcdk_torch_frame_create_host(int width, int height, int pixfmt, int align);

/**创建。*/
abcdk_torch_frame_t *abcdk_torch_frame_create_cuda(int width, int height, int pixfmt, int align);

#ifdef ABCDK_TORCH_USE_CUDA
#define abcdk_torch_frame_create abcdk_torch_frame_create_cuda
#else //ABCDK_TORCH_USE_HOST
#define abcdk_torch_frame_create abcdk_torch_frame_create_host
#endif //

__END_DECLS

#endif //ABCDK_TORCH_FRAME_H
