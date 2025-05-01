/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_DNN_POST_H
#define ABCDK_TORCH_DNN_POST_H

#include "abcdk/util/option.h"
#include "abcdk/torch/dnn.h"

__BEGIN_DECLS

/**DNN后处理环境。*/
typedef struct _abcdk_torch_dnn_post abcdk_torch_dnn_post_t;

/**释放。 */
void abcdk_torch_dnn_post_free(abcdk_torch_dnn_post_t **ctx);

/**申请。*/
abcdk_torch_dnn_post_t *abcdk_torch_dnn_post_alloc();

/**
 * 初始化。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_dnn_post_init(abcdk_torch_dnn_post_t *ctx, const char *name, abcdk_option_t *opt);

/**
 * 处理。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_dnn_post_process(abcdk_torch_dnn_post_t *ctx, int count, abcdk_torch_dnn_tensor tensor[], float score_threshold, float nms_threshold);

/**
 * 获取结果。
 *
 * @return 数量。
 */
int abcdk_torch_dnn_post_fetch(abcdk_torch_dnn_post_t *ctx, int index, int count, abcdk_torch_dnn_object_t object[]);

__END_DECLS

#endif // ABCDK_TORCH_DNN_POST_H
