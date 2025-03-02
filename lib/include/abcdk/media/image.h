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

    /**私有环境。*/
    void *private_ctx;

    /**私有环境释放。*/
    void (*private_ctx_free_cb)(void **ctx);

}abcdk_media_image_t;

/**释放。*/
void abcdk_media_image_free(abcdk_media_image_t **ctx);

/**申请。*/
abcdk_media_image_t *abcdk_media_image_alloc(uint32_t tag);

/**
 * 重置。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_media_image_reset(abcdk_media_image_t **ctx, int width, int height, int pixfmt, int align);

/**创建。*/
abcdk_media_image_t *abcdk_media_image_create(int width, int height, int pixfmt, int align);

/** 复制。 */
void abcdk_media_image_copy(abcdk_media_image_t *dst, const abcdk_media_image_t *src);

/** 复制。 */
void abcdk_media_image_copy_plane(abcdk_media_image_t *dst, int dst_plane, const uint8_t *src_data, int src_stride);

/**克隆。*/
abcdk_media_image_t *abcdk_media_image_clone(const abcdk_media_image_t *src);

/**
 * 帧图格式转换。
 *
 * @note 仅图像数据。
 *
 * @return 0 成功，< 0 失败。
 */
int abcdk_media_image_convert(abcdk_media_image_t *dst, const abcdk_media_image_t *src);

/**
 * 保存到文件(BMP)。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_media_image_save(const char *dst, const abcdk_media_image_t *src);


__END_DECLS

#endif // ABCDK_MEDIA_FRAME_H