/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_PACKET_H
#define ABCDK_TORCH_PACKET_H

#include "abcdk/torch/torch.h"

__BEGIN_DECLS

/**媒体包。*/
typedef struct _abcdk_torch_packet
{
    /**数据指针。*/
    void *data;

    /**数据长度。*/
    int size;

    /**DTS.*/
    int64_t dts;

    /**PTS.*/
    int64_t pts;

    /**缓存。*/
    abcdk_object_t *buf;

} abcdk_torch_packet_t;

/**释放。 */
void abcdk_torch_packet_free(abcdk_torch_packet_t **ctx);

/**申请。*/
abcdk_torch_packet_t *abcdk_torch_packet_alloc();

/**
 * 重置。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_torch_packet_reset(abcdk_torch_packet_t **ctx,size_t size);

/**创建。*/
abcdk_torch_packet_t *abcdk_torch_packet_create(size_t size);

__END_DECLS

#endif //ABCDK_TORCH_PACKET_H
