/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_DNN_INFER_H
#define ABCDK_XPU_DNN_INFER_H

#include "abcdk/util/option.h"
#include "abcdk/xpu/types.h"


__BEGIN_DECLS

/**推理环境.*/
typedef struct _abcdk_xpu_dnn_infer abcdk_xpu_dnn_infer_t;

/**释放. */
void abcdk_xpu_dnn_infer_free(abcdk_xpu_dnn_infer_t **ctx);

/**创建. */
abcdk_xpu_dnn_infer_t *abcdk_xpu_dnn_infer_alloc();

/**
 * 加载模型.
 *
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_dnn_infer_load_model(abcdk_xpu_dnn_infer_t *ctx, const char *file, abcdk_option_t *opt);

/**
 * 获取张量信息.
 *
 * @return 数量.
 */
int abcdk_xpu_dnn_infer_fetch_tensor(abcdk_xpu_dnn_infer_t *ctx, int count, abcdk_xpu_dnn_tensor_t tensor[]);

/**
 * 推理.
 *
 * @return 0 成功. < 失败.
 */
int abcdk_xpu_dnn_infer_forward(abcdk_xpu_dnn_infer_t *ctx, int count, abcdk_xpu_image_t *img[]);

__END_DECLS

#endif // ABCDK_XPU_DNN_INFER_H