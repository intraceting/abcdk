/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_MEDIA_FRAME_H
#define ABCDK_MEDIA_FRAME_H

#include "abcdk/media/image.h"

__BEGIN_DECLS

/**媒体帧。*/
typedef struct _abcdk_media_frame
{
    /**数据(图像)。*/
    abcdk_media_image_t *img;

    /**DTS.*/
    int64_t dts;

    /**PTS.*/
    int64_t pts;

}abcdk_media_frame_t;

/**释放。 */
void abcdk_media_frame_free(abcdk_media_frame_t **ctx);

/**申请。*/
abcdk_media_frame_t *abcdk_media_frame_alloc();

__END_DECLS

#endif //ABCDK_MEDIA_FRAME_H
