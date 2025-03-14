/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_JCODEC_H
#define ABCDK_TORCH_JCODEC_H

#include "abcdk/util/object.h"
#include "abcdk/torch/torch.h"

__BEGIN_DECLS

/**JPEG编/解码器参数。*/
typedef struct _abcdk_torch_jcodec_param
{
    /**
     * 质量。
     *
     * 1~99 值越大越清晰，占用的空间越多。
     */
    int quality;

} abcdk_torch_jcodec_param_t;

/**JPEG编/解码器。*/
typedef struct _abcdk_torch_jcodec
{
    /**标签。*/
    uint32_t tag;

    /**私有环境。*/
    void *private_ctx;

    /**私有环境释放。*/
    void (*private_ctx_free_cb)(void **ctx);

} abcdk_torch_jcodec_t;

/**释放。*/
void abcdk_torch_jcodec_free(abcdk_torch_jcodec_t **ctx);

/**申请。 */
abcdk_torch_jcodec_t *abcdk_torch_jcodec_alloc(uint32_t tag);

__END_DECLS

#endif // ABCDK_TORCH_JCODEC_H