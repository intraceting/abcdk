/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_DNN_TRACK_H
#define ABCDK_TORCH_DNN_TRACK_H

#include "abcdk/util/option.h"
#include "abcdk/torch/dnn.h"

__BEGIN_DECLS

/**DNN多目标追踪环境。*/
typedef struct _abcdk_torch_dnn_track abcdk_torch_dnn_track_t;

/**释放。 */
void abcdk_torch_dnn_track_free(abcdk_torch_dnn_track_t **ctx);

/**申请。*/
abcdk_torch_dnn_track_t *abcdk_torch_dnn_track_alloc();

/**
 * 初始化。
 * 
 * @param [in] name 追踪器名字。目前仅支持：bytetrack。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_dnn_track_init(abcdk_torch_dnn_track_t *ctx, const char *name, abcdk_option_t *opt);

/**
 * 更新追踪器。
 *
 * @return 0 成功，-1 失败。
 */
int abcdk_torch_dnn_track_update(abcdk_torch_dnn_track_t *ctx, int count, abcdk_torch_dnn_object_t object[]);

__END_DECLS

#endif // ABCDK_TORCH_DNN_TRACK_H