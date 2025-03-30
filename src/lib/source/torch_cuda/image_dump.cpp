/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/image.h"
#include "abcdk/torch/nvidia.h"

__BEGIN_DECLS

#ifdef __cuda_cuda_h__

int abcdk_torch_image_dump_cuda(const char *dst, const abcdk_torch_image_t *src)
{
    abcdk_torch_image_t *tmp_src;
    int chk;

    assert(dst != NULL && src != NULL);
    assert(src->tag == ABCDK_TORCH_TAG_CUDA || src->tag == ABCDK_TORCH_TAG_HOST);

    if (src->tag != ABCDK_TORCH_TAG_HOST)
    {
        tmp_src = abcdk_torch_image_clone_cuda(1, src);
        if (!tmp_src)
            return -1;

        chk = abcdk_torch_image_dump_cuda(dst, tmp_src);
        abcdk_torch_image_free_host(&tmp_src);

        return chk;
    }

    return abcdk_torch_image_dump_host(dst, src);
}

#endif //__cuda_cuda_h__

__END_DECLS
