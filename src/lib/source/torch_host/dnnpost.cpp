/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#include "abcdk/torch/dnnpost.h"

__BEGIN_DECLS

/*DNN后处理环境。*/
struct _abcdk_torch_dnn_post
{

}; // abcdk_torch_dnn_post_t;

void abcdk_torch_dnn_post_free(abcdk_torch_dnn_post_t **ctx)
{
}

abcdk_torch_dnn_post_t *abcdk_torch_dnn_post_alloc()
{
    return NULL;
}

int abcdk_torch_dnn_post_prepare(abcdk_torch_dnn_post_t *ctx, const char *name, abcdk_option_t *opt)
{
    return -1;
}

int abcdk_torch_dnn_post_process(abcdk_torch_dnn_post_t *ctx, int count, abcdk_torch_dnn_tensor tensor[])
{
    return -1;
}

int abcdk_torch_dnn_post_fetch(abcdk_torch_dnn_post_t *ctx, int batch, int count, abcdk_torch_dnn_object_t object[])
{
    return -1;
}

__END_DECLS