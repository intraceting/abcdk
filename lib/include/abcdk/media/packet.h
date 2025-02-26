/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_MEDIA_PACKET_H
#define ABCDK_MEDIA_PACKET_H

#include "abcdk/media/media.h"

__BEGIN_DECLS

/**媒体数据包结构。*/
typedef struct _abcdk_media_packet
{
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

    /**数据指针。*/
    void *data;

    /**数据长度。*/
    int size;

    /**解码时间。 */
    int64_t dts;

    /**播放时间。 */
    int64_t pts;
} abcdk_media_packet_t;

/**释放。*/
void abcdk_media_packet_free(abcdk_media_packet_t **ctx);

/**申请。*/
abcdk_media_packet_t *abcdk_media_packet_alloc(uint32_t tag);

/**重置。*/
int abcdk_media_packet_reset(abcdk_media_packet_t *ctx, int size);

/**创建。*/
abcdk_media_packet_t *abcdk_media_packet_create(int size);


__END_DECLS

#endif // ABCDK_MEDIA_PACKET_H