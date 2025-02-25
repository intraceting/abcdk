/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_MEDIA_FRAME_H
#define ABCDK_MEDIA_FRAME_H

#include "abcdk/util/bmp.h"
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
    int pixfmt;

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

    /**标志。*/
    uint32_t tag;
#define ABCDK_MEDIA_FRAME_TAG_HOST ABCDK_FOURCC_MKTAG('h', 'o', 's', 't')
#define ABCDK_MEDIA_FRAME_TAG_CUDA ABCDK_FOURCC_MKTAG('c', 'u', 'd', 'a')

    /**私有环境。*/
    abcdk_object_t *private_ctx;

} abcdk_media_frame_t;

/**释放。*/
void abcdk_media_frame_free(abcdk_media_frame_t **ctx);

/**创建。*/
abcdk_media_frame_t *abcdk_media_frame_alloc(uint32_t tag);

/**创建。*/
abcdk_media_frame_t *abcdk_media_frame_alloc2(int width, int height, int pixfmt, int align);

/**克隆。*/
abcdk_media_frame_t *abcdk_media_frame_clone(const abcdk_media_frame_t *src);

/**克隆。*/
abcdk_media_frame_t *abcdk_media_frame_clone2(const uint8_t *src_data[4], const int src_stride[4], int src_width, int src_height, int src_pixfmt);

/**
 * 保存到文件(BMP)。
 * 
 * @note 仅支持RGB24、BGR24、RGB32、BGR32四种格式。
 * 
 * @return 0 成功，< 0 失败。
 */
int abcdk_media_frame_save(const char *dst, const abcdk_media_frame_t *src);

__END_DECLS

#endif // ABCDK_MEDIA_FRAME_H