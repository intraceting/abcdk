/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_MEDIA_CODEC_H
#define ABCDK_MEDIA_CODEC_H

#include "abcdk/util/object.h"
#include "abcdk/media/cdcfmt.h"

__BEGIN_DECLS

/**编/解码器参数。*/
typedef struct _abcdk_media_codec_param
{
    /**帧速。 */
    int fps;

    /**宽(像素)。 */
    int width;

    /**高(像素)。 */
    int height;

    /**编码格式。 */
    int cdcfmt;

    int global_header;

    /**扩展参数。*/
    abcdk_object_t *extradata;

} abcdk_media_codec_param_t;

__END_DECLS

#endif //ABCDK_MEDIA_CODEC_H