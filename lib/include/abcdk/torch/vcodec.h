/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_TORCH_VCODEC_H
#define ABCDK_TORCH_VCODEC_H

#include "abcdk/util/object.h"
#include "abcdk/torch/torch.h"
#include "abcdk/torch/packet.h"
#include "abcdk/nvidia/frame.h"

__BEGIN_DECLS

/**视频编码常量。*/
typedef enum _abcdk_torch_vcodec_constant
{
    ABCDK_TORCH_VCODEC_NONE = -1,
#define ABCDK_TORCH_VCODEC_NONE ABCDK_TORCH_VCODEC_NONE

    ABCDK_TORCH_VCODEC_H264,
#define ABCDK_TORCH_VCODEC_H264 ABCDK_TORCH_VCODEC_H264

    ABCDK_TORCH_VCODEC_HEVC,
#define ABCDK_TORCH_VCODEC_HEVC ABCDK_TORCH_VCODEC_HEVC

    ABCDK_TORCH_VCODEC_MJPEG,
#define ABCDK_TORCH_VCODEC_MJPEG ABCDK_TORCH_VCODEC_MJPEG

    ABCDK_TORCH_VCODEC_MPEG1VIDEO,
#define ABCDK_TORCH_VCODEC_MPEG1VIDEO ABCDK_TORCH_VCODEC_MPEG1VIDEO

    ABCDK_TORCH_VCODEC_MPEG2VIDEO,
#define ABCDK_TORCH_VCODEC_MPEG2VIDEO ABCDK_TORCH_VCODEC_MPEG2VIDEO

    ABCDK_TORCH_VCODEC_MPEG4,
#define ABCDK_TORCH_VCODEC_MPEG4 ABCDK_TORCH_VCODEC_MPEG4

    ABCDK_TORCH_VCODEC_VC1,
#define ABCDK_TORCH_VCODEC_VC1 ABCDK_TORCH_VCODEC_VC1

    ABCDK_TORCH_VCODEC_VP8,
#define ABCDK_TORCH_VCODEC_VP8 ABCDK_TORCH_VCODEC_VP8

    ABCDK_TORCH_VCODEC_VP9,
#define ABCDK_TORCH_VCODEC_VP9 ABCDK_TORCH_VCODEC_VP9

    ABCDK_TORCH_VCODEC_WMV3,
#define ABCDK_TORCH_VCODEC_WMV3 ABCDK_TORCH_VCODEC_WMV3

} abcdk_torch_vcodec_constant_t;

/**视频编/解码器参数。*/
typedef struct _abcdk_torch_vcodec_param
{
    /** 编码格式 */
    uint32_t format;

    /** 宽度（像素） */
    uint32_t width;

    /** 高度（像素） */
    uint32_t height;

    /**
     * 编码 Profile.
     * H.264: 66 (Baseline), 77 (Main), 100 (High)
     * H.265: 1 (Main), 2 (Main10), 3 (Main12)
     */
    uint32_t profile;

    /**
     * 编码 Level
     * H.264 典型值: 40 (4.0), 41 (4.1), 50 (5.0), 51 (5.1)
     * H.265 典型值: 51 (5.1), 62 (6.2)
     */
    uint32_t level;

    /** 目标比特率（单位：bps）. */
    uint32_t bitrate;

    /**
     * 峰值比特率（仅在 VBR 模式下生效，单位：bps）.
     * 在 CBR 模式下，该参数无效
     */
    uint32_t peak_bitrate;

    /** 是否启用无损编码（true = 启用，false = 关闭）. */
    uint8_t enableLossless;

    /**
     * 是否启用 VBR（可变比特率）模式.
     * false = CBR（固定比特率），true = VBR（可变比特率）
     */
    uint8_t mode_vbr;

    /**
     * 是否在 IDR 帧插入 SPS/PPS
     * true = 在每个 IDR 帧前插入 SPS/PPS，适用于流式编码
     */
    uint8_t insert_spspps_idr;

    /** I 帧间隔（每隔多少帧插入 I 帧） */
    uint32_t iframe_interval;

    /** IDR 帧间隔（每隔多少帧插入 IDR 帧） */
    uint32_t idr_interval;

    /**
     * 帧率分子（如 30fps，则 fps_n = 30, fps_d = 1）
     * 帧率计算方式: fps = fps_n / fps_d
     */
    uint32_t fps_n;

    /** 帧率分母 */
    uint32_t fps_d;

    /**
     * 捕获帧数量（用于缓冲或并行处理）
     * 设为 -1 时表示默认值
     */
    int capture_num;

    /**
     * 最大 B 帧数量
     * H.264/H.265 编码中 B 帧用于提高压缩率
     * 典型值: 0（无 B 帧）, 3（平衡模式）, 4-5（高压缩率）
     */
    uint32_t max_b_frames;

    /**
     * 参考帧数量
     * H.264: 1-4
     * H.265: 1-8（更高参考帧数可提高压缩效率）
     */
    uint32_t refs;

    /**
     * 最大量化参数（QP，值越大质量越低）
     * 取值范围 0-51（H.264），0-63（H.265）
     */
    uint32_t qmax;

    /**
     * 最小量化参数（QP，值越小质量越高）
     * 取值范围 0-51（H.264），0-63（H.265）
     */
    uint32_t qmin;

    /**
     * 硬件编码预设类型
     * 0 = 默认, 1 = 低延迟, 2 = 高质量, 3 = 高性能
     */
    uint32_t hw_preset_type;

    /**扩展参数指针。*/
    const uint8_t *ext_data;

    /**扩展参数长度。*/
    uint32_t ext_size;

} abcdk_torch_vcodec_param_t;

/** 媒体视频编/解码器。*/
typedef struct _abcdk_torch_vcodec
{
    /**标签。*/
    uint32_t tag;

    /**私有环境。*/
    void *private_ctx;

    /**私有环境释放。*/
    void (*private_ctx_free_cb)(void **ctx);

} abcdk_torch_vcodec_t;

/**转为ffmpeg类型。*/
int abcdk_torch_vcodec_convert_to_ffmpeg(int format);

/**从ffmpeg类型转。*/
int abcdk_torch_vcodec_convert_from_ffmpeg(int format);

/**释放。*/
void abcdk_torch_vcodec_free(abcdk_torch_vcodec_t **ctx);

/**申请。 */
abcdk_torch_vcodec_t *abcdk_torch_vcodec_alloc(uint32_t tag);

__END_DECLS

#endif // ABCDK_TORCH_VCODEC_H