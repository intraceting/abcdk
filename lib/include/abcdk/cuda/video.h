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

/**释放。*/
void abcdk_cuda_video_destroy(abcdk_cuda_video_t **ctx);

/**创建。*/
abcdk_cuda_video_t *abcdk_cuda_video_create(int encode, abcdk_option_t *cfg);

/**同步。*/
int abcdk_cuda_video_sync(abcdk_cuda_video_t *ctx,AVCodecContext *opt);

/**编码。 */
int abcdk_cuda_video_encode(abcdk_cuda_video_t *ctx,AVPacket **dst, const AVFrame *src);

/**解码。 */
int abcdk_cuda_video_decode(abcdk_cuda_video_t *ctx,AVFrame **dst, const AVPacket *src);

__END_DECLS

#endif //AVCODEC_AVCODEC_H
#endif //AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__

#endif // ABCDK_CUDA_VIDEO_H