/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_MEDIA_IMAGE_H
#define ABCDK_MEDIA_IMAGE_H

#include "abcdk/media/media.h"
#include "abcdk/media/imgutil.h"
#include "abcdk/ffmpeg/swscale.h"

__BEGIN_DECLS

/**媒体图像结构。*/
typedef struct _abcdk_media_image
{
    /**图层指针。 */
    uint8_t *data[4];

    /**图层步长。 */
    int stride[4];

    /**图像格式。 */
    int pixfmt;

    /**宽(像素)。*/
    int width;

    /**高(像素)。*/
    int height;

    /**标签。*/
    uint32_t tag;

    /**缓存指针。*/
    void *buf_ptr;

    /**缓存长度。*/
    int buf_size;

    /**缓存释放。*/
    void (*buffer_free_cb)(void **ptr, int size);

    /**缓存申请。*/
    int (*buffer_alloc_cb)(void **ptr, int size);

}abcdk_media_image_t;

/**释放。*/
void abcdk_media_image_free(abcdk_media_image_t **ctx);

/**申请。*/
abcdk_media_image_t *abcdk_media_image_alloc(uint32_t tag);

/**重置。*/
int abcdk_media_image_reset(abcdk_media_image_t *ctx, int width, int height, int pixfmt, int align);

/**创建。*/
abcdk_media_image_t *abcdk_media_image_create(int width, int height, int pixfmt, int align);

/**
 * 复制。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_media_image_copy(abcdk_media_image_t *dst, const abcdk_media_image_t *src);

/**克隆。*/
abcdk_media_image_t *abcdk_media_image_clone(const abcdk_media_image_t *src);

/**
 * 保存到文件(BMP)。
 * 
 * @note 仅支持RGB24、BGR24、RGB32、BGR32四种格式。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_media_image_save(const char *dst, const abcdk_media_image_t *src);

/**
 * 帧图格式转换。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_media_image_convert(abcdk_media_image_t *dst, const abcdk_media_image_t *src);

__END_DECLS

#endif // ABCDK_MEDIA_FRAME_H