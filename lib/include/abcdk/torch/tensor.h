/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_TENSOR_H
#define ABCDK_TORCH_TENSOR_H

#include "abcdk/torch/torch.h"
#include "abcdk/torch/tenutil.h"

__BEGIN_DECLS

/**简单的张量。*/
typedef struct _abcdk_torch_tensor
{
    /**数据指针。*/
    uint8_t *data;

    /**格式。*/
    int format;

    /**块数量。*/
    size_t block;

    /**宽(每块)。*/
    size_t width;
    
    /**宽步长(字节)。*/
    size_t stride;

    /**高(每块)。*/
    size_t height;

    /**深(每块)。*/
    size_t depth;

    /**单元尺寸(字节)。*/
    size_t cell;

    /**标签。*/
    uint32_t tag;

    /**私有环境。*/
    void *private_ctx;

    /**私有环境释放。*/
    void (*private_ctx_free_cb)(void **ctx);
} abcdk_torch_tensor_t;


/**释放。*/
void abcdk_torch_tensor_free(abcdk_torch_tensor_t **ctx);

/**申请。*/
abcdk_torch_tensor_t *abcdk_torch_tensor_alloc(uint32_t tag);

/**
 * 重置。
 * 
 * @return 0 成功，< 0 失败。
*/
int abcdk_torch_tensor_reset(abcdk_torch_tensor_t **ctx, int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align);

/**创建。*/
abcdk_torch_tensor_t *abcdk_torch_tensor_create(int format, size_t block, size_t width, size_t height, size_t depth, size_t cell, size_t align);

/** 复制。
 */
void abcdk_torch_tensor_copy(abcdk_torch_tensor_t *dst, const abcdk_torch_tensor_t *src);

/**
 * 克隆。
 */
abcdk_torch_tensor_t *abcdk_torch_tensor_clone(const abcdk_torch_tensor_t *src);


__END_DECLS

#endif // ABCDK_TORCH_TENSOR_H