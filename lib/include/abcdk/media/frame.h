/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_MEDIA_FRAME_H
#define ABCDK_MEDIA_FRAME_H

#include "abcdk/media/media.h"
#include "abcdk/media/image.h"
#include "abcdk/ffmpeg/swscale.h"

__BEGIN_DECLS

/**媒体帧图结构。*/
typedef struct _abcdk_media_frame
{
    /**私有环境。*/
    void *private_ctx;

    /**解码时间。 */
    int64_t dts;

    /**播放时间。 */
    int64_t pts;

    /**标签。*/
    uint32_t tag;

    /**私有环境释放。*/
    void (*private_ctx_free_cb)(void **ctx);

    /**图像上传。*/
    int (*image_upload_cb)(void *ctx, const abcdk_media_image_t *src);

    /**图像下载。*/
    int (*image_download_cb)(void *ctx, abcdk_media_image_t **dst);

}abcdk_media_frame_t;


/**释放。*/
void abcdk_media_frame_free(abcdk_media_frame_t **ctx);

/**申请。*/
abcdk_media_frame_t *abcdk_media_frame_alloc(uint32_t tag);

/**
 * 图像上传。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_media_frame_image_upload(abcdk_media_frame_t *ctx, const abcdk_media_image_t *src);

/**
 * 图像下载。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_media_frame_image_download(abcdk_media_frame_t *ctx, abcdk_media_image_t **dst);

__END_DECLS

#endif // ABCDK_MEDIA_FRAME_H