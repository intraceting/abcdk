/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_DNN_TRACK_H
#define ABCDK_XPU_DNN_TRACK_H

#include "abcdk/xpu/types.h"

__BEGIN_DECLS

/**追踪环境.*/
typedef struct _abcdk_xpu_dnn_track abcdk_xpu_dnn_track_t;

/**释放. */
void abcdk_xpu_dnn_track_free(abcdk_xpu_dnn_track_t **ctx);

/**创建. */
abcdk_xpu_dnn_track_t *abcdk_xpu_dnn_track_alloc();

/**
 * 初始化.
 * 
 * @param [in] name 追踪器名字. 目前仅支持：bytetrack.
 *
 * @return 0 成功, -1 失败.
 */
int abcdk_torch_dnn_track_init(abcdk_xpu_dnn_track_t *ctx, const char *name, abcdk_option_t *opt);

/**
 * 更新追踪器.
 *
 * @return 0 成功, -1 失败.
 */
int abcdk_xpu_dnn_track_update(abcdk_xpu_dnn_track_t *ctx, int count, abcdk_xpu_dnn_object_t object[]);

__END_DECLS

#endif // ABCDK_XPU_DNN_TRACK_H