/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_VENC_H
#define ABCDK_XPU_VENC_H

#include "abcdk/xpu/types.h"


__BEGIN_DECLS

/**编码环境.*/
typedef struct _abcdk_xpu_venc abcdk_xpu_venc_t;

/**释放. */
void abcdk_xpu_venc_free(abcdk_xpu_venc_t **ctx);

/**创建. */
abcdk_xpu_venc_t *abcdk_xpu_venc_alloc();

/**
 * 设置.
 * 
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_venc_setup(abcdk_xpu_venc_t *ctx, const abcdk_xpu_vcodec_params_t *params);

/**
 * 获取参数.
 * 
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_venc_get_params(abcdk_xpu_venc_t *ctx, abcdk_xpu_vcodec_params_t *params);

/**
 * 接收. 
 * 
 * @return > 0 有, 0 无, < 0 出错.
*/
int abcdk_xpu_venc_recv_packet(abcdk_xpu_venc_t *ctx ,abcdk_object_t **dst, int64_t *ts);

/**
 * 发送.
 * 
 * @param [in] src 图像. NULL(0) 末尾.
 * 
 * @return > 0 成功, 0 忙, < 0 出错.
 */
int abcdk_xpu_venc_send_frame(abcdk_xpu_venc_t *ctx ,const abcdk_xpu_image_t *src,int64_t ts);


__END_DECLS

#endif //ABCDK_XPU_VENC_H