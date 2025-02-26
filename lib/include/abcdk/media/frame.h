/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_MEDIA_FRAME_H
#define ABCDK_MEDIA_FRAME_H

#include "abcdk/ffmpeg/swscale.h"
#include "abcdk/media/media.h"
#include "abcdk/media/imgutil.h"

__BEGIN_DECLS

/**媒体帧图结构。*/
typedef struct _abcdk_media_frame
{
    /**图层指针。 */
    uint8_t *data[4];

    /**图层步长。 */
    int stride[4];

    /**像素格式。 */
    int pixfmt;

    /**宽。*/
    int width;

    /**高。*/
    int height;

    /**缓存。*/
    abcdk_object_t *buf;

    /**标签。*/
    uint32_t tag;

    /**解码时间。 */
    int64_t dts;

    /**播放时间。 */
    int64_t pts;
}abcdk_media_frame_t;

/**释放。*/
void abcdk_media_frame_free(abcdk_media_frame_t **ctx);

/**申请。*/
abcdk_media_frame_t *abcdk_media_frame_alloc(uint32_t tag);

/**重置。*/
int abcdk_media_frame_reset(abcdk_media_frame_t *ctx, int width, int height, int pixfmt, int align);

/**创建。*/
abcdk_media_frame_t *abcdk_media_frame_create(int width, int height, int pixfmt, int align);

/**
 * 保存到文件(BMP)。
 * 
 * @note 仅支持RGB24、BGR24、RGB32、BGR32四种格式。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_media_frame_save(const char *dst, const abcdk_media_frame_t *src);

/**
 * 复制。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_media_frame_copy(abcdk_media_frame_t *dst, const abcdk_media_frame_t *src);

/**克隆。*/
abcdk_media_frame_t *abcdk_media_frame_clone(const abcdk_media_frame_t *src);

/**克隆。*/
abcdk_media_frame_t *abcdk_media_frame_clone2(const uint8_t *src_data[4], const int src_stride[4], int src_width, int src_height, int src_pixfmt);

/**
 * 帧图格式转换。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_media_frame_convert(abcdk_media_frame_t *dst, const abcdk_media_frame_t *src);

__END_DECLS

#endif // ABCDK_MEDIA_FRAME_H