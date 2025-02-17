/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_CUDA_JPEG_H
#define ABCDK_CUDA_JPEG_H

#include "abcdk/util/option.h"
#include "abcdk/util/object.h"
#include "abcdk/cuda/cuda.h"
#include "abcdk/cuda/avutil.h"

#ifdef __cuda_cuda_h__
#ifdef AVUTIL_AVUTIL_H

__BEGIN_DECLS

/**JPEG编/解码器。*/
typedef struct _abcdk_cuda_jpeg abcdk_cuda_jpeg_t;

/**释放。*/
void abcdk_cuda_jpeg_destroy(abcdk_cuda_jpeg_t **ctx);

/**创建。*/
abcdk_cuda_jpeg_t *abcdk_cuda_jpeg_create(int encode, abcdk_option_t *cfg);

/**编码。 */
abcdk_object_t *abcdk_cuda_jpeg_encode(abcdk_cuda_jpeg_t *ctx,const AVFrame *src);

/**
 * 编码。 
 * 
 * @return 0 成功，< 0  失败。
*/
int abcdk_cuda_jpeg_encode_to_file(abcdk_cuda_jpeg_t *ctx, const char *dst, const AVFrame *src);

/**解码。 */
AVFrame *abcdk_cuda_jpeg_decode(abcdk_cuda_jpeg_t *ctx,const void *src,int src_size);

/**解码。 */
AVFrame *abcdk_cuda_jpeg_decode_from_file(abcdk_cuda_jpeg_t *ctx,const void *src);

/**保存。*/
int abcdk_cuda_jpeg_save(const char *dst, const AVFrame *src);

/**加载。*/
AVFrame *abcdk_cuda_jpeg_load(const char *src);

__END_DECLS

#endif //AVUTIL_AVUTIL_H
#endif //__cuda_cuda_h__

#endif // ABCDK_CUDA_JPEG_H