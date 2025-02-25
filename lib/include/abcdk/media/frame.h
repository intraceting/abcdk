/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_MEDIA_FRAME_H
#define ABCDK_MEDIA_FRAME_H

#include "abcdk/media/imgutil.h"

__BEGIN_DECLS

/**媒体帧图结构。*/
typedef struct _abcdk_media_frame
{
    /**图像数据指针。 */
    uint8_t *data[4];
    /**图像数据步长(字节)。 */
    int stride[4];

    /**图像格式。 */
    int format;

    /**宽(像素)。 */
    int width;

    /**高(像素)。 */
    int height;

    /**解码时间。*/
    int64_t dts;

    /**播放时间。*/
    int64_t pts;

    /**缓存。*/
    abcdk_object_t *buf;

    /**硬件环境。*/
    abcdk_object_t *hw_ctx;

} abcdk_media_frame_t;

/**释放。*/
void abcdk_media_frame_free(abcdk_media_frame_t **ctx);

/**创建。*/
abcdk_media_frame_t *abcdk_media_frame_alloc();

/**创建。*/
abcdk_media_frame_t *abcdk_media_frame_alloc2(int width, int height, int pixfmt, int align);

/**克隆。*/
abcdk_media_frame_t *abcdk_media_frame_clone(const abcdk_media_frame_t *src);

/**克隆。*/
abcdk_media_frame_t *abcdk_media_frame_clone2(const uint8_t *src_data[4], const int src_stride[4], int src_width, int src_height, int src_pixfmt);

__END_DECLS

#endif // ABCDK_MEDIA_FRAME_H