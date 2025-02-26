/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_MEDIA_JCODEC_H
#define ABCDK_MEDIA_JCODEC_H

#include "abcdk/util/object.h"
#include "abcdk/media/media.h"

__BEGIN_DECLS

/** 媒体JPEG编/解码器参数。*/
typedef struct _abcdk_media_jcodec_param
{
    /**
     * 质量。
     * 
     * 1~99 值越大越清晰，占用的空间越多。
    */
   int quality;

}abcdk_media_jcodec_param_t;

/** 媒体JPEG编/解码器。*/
typedef struct _abcdk_media_jcodec
{
    /**标签。*/
    uint32_t tag;

    /**私有环境。*/
    void *private_ctx;

    /**私有环境释放。*/
    int (*private_ctx_free_cb)(void **ctx, uint8_t encoder);

    /** 是否为编码器（true = 是，false = 否）。 */
    uint8_t encoder;

} abcdk_media_jcodec_t;

/**释放。*/
void abcdk_media_jcodec_free(abcdk_media_jcodec_t **ctx);

/**申请。 */
abcdk_media_jcodec_t *abcdk_media_jcodec_alloc(uint32_t tag);

__END_DECLS

#endif // ABCDK_MEDIA_JCODEC_H