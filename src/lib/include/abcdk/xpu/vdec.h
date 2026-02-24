/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_VDEC_H
#define ABCDK_XPU_VDEC_H

#include "abcdk/xpu/image.h"
#include "abcdk/ffmpeg/decoder.h"

__BEGIN_DECLS

/**解码环境.*/
typedef struct _abcdk_xpu_vdec abcdk_xpu_vdec_t;

/**释放. */
void abcdk_xpu_vdec_free(abcdk_xpu_vdec_t **ctx);

/**创建. */
abcdk_xpu_vdec_t *abcdk_xpu_vdec_alloc();

/**
 * 设置.
 * 
 * @return 0 成功, < 0 失败.
 */
int abcdk_xpu_vdec_setup(abcdk_xpu_vdec_t *ctx, const abcdk_xpu_vcodec_params_t *params);

/**
 * 发送. 
 * 
 * @param [in] src_data 数据. NULL(0) 末尾.
 * @param [in] src_size 长度. <= 0 末尾.
 * 
 * @return > 0 成功, 0 忙, < 0 出错.
*/
int abcdk_xpu_vdec_send_packet(abcdk_xpu_vdec_t *ctx, const void *src_data, size_t src_size, int64_t ts);

/**
 * 接收.
 * 
 * @return > 0 有, 0 无, < 0 出错.
 */
int abcdk_xpu_vdec_recv_frame(abcdk_xpu_vdec_t *ctx, abcdk_xpu_image_t **dst, int64_t *ts);

__END_DECLS

#endif // ABCDK_XPU_VDEC_H