/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2025 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_XPU_TYPES_H
#define ABCDK_XPU_TYPES_H

#include "abcdk/util/defs.h"
#include "abcdk/util/option.h"

__BEGIN_DECLS

/**hardware acceleration.*/
typedef enum _abcdk_xpu_hwaccel
{
    /**NONE. */
    ABCDK_XPU_HWACCEL_NONE = 0,
#define ABCDK_XPU_HWACCEL_NONE ABCDK_XPU_HWACCEL_NONE

    /**NVIDIA. */
    ABCDK_XPU_HWACCEL_NVIDIA = 1,
#define ABCDK_XPU_HWACCEL_NVIDIA ABCDK_XPU_HWACCEL_NVIDIA

    /**SOPHON. */
    ABCDK_XPU_HWACCEL_SOPHON = 2,
#define ABCDK_XPU_HWACCEL_SOPHON ABCDK_XPU_HWACCEL_SOPHON

    /**ROCKCHIP. */
    ABCDK_XPU_HWACCEL_ROCKCHIP = 3,
#define ABCDK_XPU_HWACCEL_ROCKCHIP ABCDK_XPU_HWACCEL_ROCKCHIP

} abcdk_xpu_hwaccel_t;

/**插值算法.*/
typedef enum _abcdk_xpu_inter
{
    /** nearest neighbor interpolation. */
    ABCDK_XPU_INTER_NEAREST = 0,
#define ABCDK_XPU_INTER_NEAREST ABCDK_XPU_INTER_NEAREST

    /** bilinear interpolation. */
    ABCDK_XPU_INTER_LINEAR = 1,
#define ABCDK_XPU_INTER_LINEAR ABCDK_XPU_INTER_LINEAR

    /** bicubic interpolation. */
    ABCDK_XPU_INTER_CUBIC = 2,
#define ABCDK_XPU_INTER_CUBIC ABCDK_XPU_INTER_CUBIC

} abcdk_xpu_inter_t;

/**像素格式.*/
typedef enum _abcdk_xpu_pixfmt
{
    ABCDK_XPU_PIXFMT_NONE = -1,
#define ABCDK_XPU_PIXFMT_NONE ABCDK_XPU_PIXFMT_NONE

    ABCDK_XPU_PIXFMT_YUV420P = 1, ///< planar YUV 4:2:0, 12bpp, (1 Cr & Cb sample per 2x2 Y samples)
#define ABCDK_XPU_PIXFMT_YUV420P ABCDK_XPU_PIXFMT_YUV420P

    ABCDK_XPU_PIXFMT_YUV420P9,
#define ABCDK_XPU_PIXFMT_YUV420P9 ABCDK_XPU_PIXFMT_YUV420P9

    ABCDK_XPU_PIXFMT_YUV420P10,
#define ABCDK_XPU_PIXFMT_YUV420P10 ABCDK_XPU_PIXFMT_YUV420P10

    ABCDK_XPU_PIXFMT_YUV420P12,
#define ABCDK_XPU_PIXFMT_YUV420P12 ABCDK_XPU_PIXFMT_YUV420P12

    ABCDK_XPU_PIXFMT_YUV420P14,
#define ABCDK_XPU_PIXFMT_YUV420P14 ABCDK_XPU_PIXFMT_YUV420P14

    ABCDK_XPU_PIXFMT_YUV420P16,
#define ABCDK_XPU_PIXFMT_YUV420P16 ABCDK_XPU_PIXFMT_YUV420P16

    ABCDK_XPU_PIXFMT_YUV422P = 20, ///< planar YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
#define ABCDK_XPU_PIXFMT_YUV422P ABCDK_XPU_PIXFMT_YUV422P

    ABCDK_XPU_PIXFMT_YUV422P9,
#define ABCDK_XPU_PIXFMT_YUV422P9 ABCDK_XPU_PIXFMT_YUV422P9

    ABCDK_XPU_PIXFMT_YUV422P10,
#define ABCDK_XPU_PIXFMT_YUV422P10 ABCDK_XPU_PIXFMT_YUV422P10

    ABCDK_XPU_PIXFMT_YUV422P12,
#define ABCDK_XPU_PIXFMT_YUV422P12 ABCDK_XPU_PIXFMT_YUV422P12

    ABCDK_XPU_PIXFMT_YUV422P14,
#define ABCDK_XPU_PIXFMT_YUV422P14 ABCDK_XPU_PIXFMT_YUV422P14

    ABCDK_XPU_PIXFMT_YUV422P16,
#define ABCDK_XPU_PIXFMT_YUV422P16 ABCDK_XPU_PIXFMT_YUV422P16

    ABCDK_XPU_PIXFMT_YUV444P = 30, ///< planar YUV 4:4:4, 24bpp, (1 Cr & Cb sample per 1x1 Y samples)
#define ABCDK_XPU_PIXFMT_YUV444P ABCDK_XPU_PIXFMT_YUV444P

    ABCDK_XPU_PIXFMT_YUV444P9,
#define ABCDK_XPU_PIXFMT_YUV444P9 ABCDK_XPU_PIXFMT_YUV444P9

    ABCDK_XPU_PIXFMT_YUV444P10,
#define ABCDK_XPU_PIXFMT_YUV444P10 ABCDK_XPU_PIXFMT_YUV444P10

    ABCDK_XPU_PIXFMT_YUV444P12,
#define ABCDK_XPU_PIXFMT_YUV444P12 ABCDK_XPU_PIXFMT_YUV444P12

    ABCDK_XPU_PIXFMT_YUV444P14,
#define ABCDK_XPU_PIXFMT_YUV444P14 ABCDK_XPU_PIXFMT_YUV444P14

    ABCDK_XPU_PIXFMT_YUV444P16,
#define ABCDK_XPU_PIXFMT_YUV444P16 ABCDK_XPU_PIXFMT_YUV444P16

    ABCDK_XPU_PIXFMT_NV12 = 40, ///< planar YUV 4:2:0, 12bpp, 1 plane for Y and 1 plane for the UV components, which are interleaved (first byte U and the following byte V)
#define ABCDK_XPU_PIXFMT_NV12 ABCDK_XPU_PIXFMT_NV12

    ABCDK_XPU_PIXFMT_P016, ///< like NV12, with 16bpp per component
#define ABCDK_XPU_PIXFMT_P016 ABCDK_XPU_PIXFMT_P016

    ABCDK_XPU_PIXFMT_NV16, ///< interleaved chroma YUV 4:2:2, 16bpp, (1 Cr & Cb sample per 2x1 Y samples)
#define ABCDK_XPU_PIXFMT_NV16 ABCDK_XPU_PIXFMT_NV16

    ABCDK_XPU_PIXFMT_NV21, ///< as above, but U and V bytes are swapped
#define ABCDK_XPU_PIXFMT_NV21 ABCDK_XPU_PIXFMT_NV21

    ABCDK_XPU_PIXFMT_NV24, ///< planar YUV 4:4:4, 24bpp, 1 plane for Y and 1 plane for the UV components, which are interleaved (first byte U and the following byte V)
#define ABCDK_XPU_PIXFMT_NV24 ABCDK_XPU_PIXFMT_NV24

    ABCDK_XPU_PIXFMT_NV42, ///< as above, but U and V bytes are swapped
#define ABCDK_XPU_PIXFMT_NV42 ABCDK_XPU_PIXFMT_NV42

    ABCDK_XPU_PIXFMT_GRAY8 = 50,
#define ABCDK_XPU_PIXFMT_GRAY8 ABCDK_XPU_PIXFMT_GRAY8

    ABCDK_XPU_PIXFMT_GRAY16,
#define ABCDK_XPU_PIXFMT_GRAY16 ABCDK_XPU_PIXFMT_GRAY16

    ABCDK_XPU_PIXFMT_GRAYF32,
#define ABCDK_XPU_PIXFMT_GRAYF32 ABCDK_XPU_PIXFMT_GRAYF32

    ABCDK_XPU_PIXFMT_RGB24 = 60, ///< packed RGB 8:8:8, 24bpp, RGBRGB...
#define ABCDK_XPU_PIXFMT_RGB24 ABCDK_XPU_PIXFMT_RGB24

    ABCDK_XPU_PIXFMT_BGR24, ///< packed RGB 8:8:8, 24bpp, BGRBGR...
#define ABCDK_XPU_PIXFMT_BGR24 ABCDK_XPU_PIXFMT_BGR24

    ABCDK_XPU_PIXFMT_RGB32,
#define ABCDK_XPU_PIXFMT_RGB32 ABCDK_XPU_PIXFMT_RGB32

    ABCDK_XPU_PIXFMT_BGR32,
#define ABCDK_XPU_PIXFMT_BGR32 ABCDK_XPU_PIXFMT_BGR32

} abcdk_xpu_pixfmt_t;


/**比较操作码. */
typedef enum _abcdk_xpu_cmpop
{
    ABCDK_XPU_CMPOP_LESS = 4,
#define ABCDK_XPU_CMPOP_LESS ABCDK_XPU_CMPOP_LESS

    ABCDK_XPU_CMPOP_LESS_EQ = 3,
#define ABCDK_XPU_CMPOP_LESS_EQ ABCDK_XPU_CMPOP_LESS_EQ

    ABCDK_XPU_CMPOP_EQ = 0,
#define ABCDK_XPU_CMPOP_EQ ABCDK_XPU_CMPOP_EQ

    ABCDK_XPU_CMPOP_GREATER_EQ = 1,
#define ABCDK_XPU_CMPOP_GREATER_EQ ABCDK_XPU_CMPOP_GREATER_EQ

    ABCDK_XPU_CMPOP_GREATER = 2
#define ABCDK_XPU_CMPOP_GREATER ABCDK_XPU_CMPOP_GREATER
} abcdk_xpu_cmpop_t;

/**点.*/
typedef struct _abcdk_xpu_point
{
    int x;
    int y;
} abcdk_xpu_point_t;

/**尺寸.*/
typedef struct _abcdk_xpu_size2i
{
    int width;
    int height;
} abcdk_xpu_size2i_t;

/**尺寸.*/
typedef abcdk_xpu_size2i_t abcdk_xpu_size_t;

/**尺寸.*/
typedef struct _abcdk_xpu_size2l
{
    int64_t width;
    int64_t height;
} abcdk_xpu_size2l_t;

/**尺寸.*/
typedef struct _abcdk_xpu_size2f
{
    float width;
    float height;
} abcdk_xpu_size2f_t;

/**尺寸.*/
typedef struct _abcdk_xpu_size2d
{
    double width;
    double height;
} abcdk_xpu_size2d_t;

/**矩形.*/
typedef struct _abcdk_xpu_rect
{
    int x;
    int y;
    int width;
    int height;
} abcdk_xpu_rect_t;

/**多边形.*/
typedef struct _abcdk_xpu_polygon
{
    int nb;
    abcdk_xpu_point_t pt[11];
} abcdk_xpu_polygon_t;

/**维度.*/
typedef struct _abcdk_xpu_dims
{
    int nb;
    int d[8];
} abcdk_xpu_dims_t;

/**标量.*/
typedef union _abcdk_xpu_scalar
{
    int8_t i8[9];
    uint8_t u8[9];
    int16_t i16[9];
    uint16_t u16[9];
    int32_t i32[9];
    uint32_t u32[9];
    int64_t i64[9];
    uint64_t u64[9];
    float f32[9];  // see IEEE 754.
    double f64[9]; // see IEEE 754.
} abcdk_xpu_scalar_t;

/**矩阵.*/
typedef union _abcdk_xpu_matrix_3x3
{
    int8_t i8[3][3];
    uint8_t u8[3][3];
    int16_t i16[3][3];
    uint16_t u16[3][3];
    int32_t i32[3][3];
    uint32_t u32[3][3];
    int64_t i64[3][3];
    uint64_t u64[3][3];
    float f32[3][3];  // see IEEE 754.
    double f64[3][3]; // see IEEE 754.
} abcdk_xpu_matrix_3x3_t;


/**视频编码常量.*/
typedef enum _abcdk_xpu_vcodec_id
{
    ABCDK_XPU_VCODEC_ID_NONE = -1,
#define ABCDK_XPU_VCODEC_ID_NONE ABCDK_XPU_VCODEC_ID_NONE

    ABCDK_XPU_VCODEC_ID_MJPEG = 1,
#define ABCDK_XPU_VCODEC_ID_MJPEG ABCDK_XPU_VCODEC_ID_MJPEG

    ABCDK_XPU_VCODEC_ID_MPEG1VIDEO = 10,
#define ABCDK_XPU_VCODEC_ID_MPEG1VIDEO ABCDK_XPU_VCODEC_ID_MPEG1VIDEO

    ABCDK_XPU_VCODEC_ID_MPEG2VIDEO,
#define ABCDK_XPU_VCODEC_ID_MPEG2VIDEO ABCDK_XPU_VCODEC_ID_MPEG2VIDEO

    ABCDK_XPU_VCODEC_ID_MPEG4 = 20,
#define ABCDK_XPU_VCODEC_ID_MPEG4 ABCDK_XPU_VCODEC_ID_MPEG4

    ABCDK_XPU_VCODEC_ID_H264 = 30,
#define ABCDK_XPU_VCODEC_ID_H264 ABCDK_XPU_VCODEC_ID_H264

    ABCDK_XPU_VCODEC_ID_HEVC = 40,
#define ABCDK_XPU_VCODEC_ID_HEVC ABCDK_XPU_VCODEC_ID_HEVC
#define ABCDK_XPU_VCODEC_ID_H265 ABCDK_XPU_VCODEC_ID_HEVC 

    ABCDK_XPU_VCODEC_ID_VC1 = 50,
#define ABCDK_XPU_VCODEC_ID_VC1 ABCDK_XPU_VCODEC_ID_VC1

    ABCDK_XPU_VCODEC_ID_VP8,
#define ABCDK_XPU_VCODEC_ID_VP8 ABCDK_XPU_VCODEC_ID_VP8

    ABCDK_XPU_VCODEC_ID_VP9,
#define ABCDK_XPU_VCODEC_ID_VP9 ABCDK_XPU_VCODEC_ID_VP9

} abcdk_xpu_vcodec_id_t;

/**视频编(解)码参数.*/
typedef struct _abcdk_xpu_vcodec_params
{
    /** 编码格式. */
    uint32_t format;

    /** 宽度(像素). */
    uint32_t width;

    /** 高度(像素). */
    uint32_t height;

    /**
     * 编码 Profile.
     * H.264: 66 (Baseline), 77 (Main), 100 (High)
     * H.265: 1 (Main), 2 (Main10), 3 (Main12)
     */
    uint32_t profile;

    /**
     * 编码 Level.
     * H.264 典型值: 40 (4.0), 41 (4.1), 50 (5.0), 51 (5.1)
     * H.265 典型值: 51 (5.1), 62 (6.2)
     */
    uint32_t level;

    /** 目标比特率(单位：bps). */
    uint32_t bitrate;

    /**
     * 最大比特率(仅在 VBR 模式下生效, 单位：bps).
     * 在 CBR 模式下, 该参数无效.
     */
    uint32_t max_bitrate;

    /**
     * 是否启用 VBR(可变比特率)模式.
     * false = CBR(固定比特率), true = VBR(可变比特率)
     */
    uint8_t mode_vbr;

    /**
     * 是否在 IDR 帧插入 SPS/PPS.
     * true = 在每个 IDR 帧前插入 SPS/PPS, 适用于流式编码
     */
    uint8_t insert_spspps_idr;

    /** I 帧间隔(每隔多少帧插入 I 帧). */
    uint32_t iframe_interval;

    /** IDR 帧间隔(每隔多少帧插入 IDR 帧). */
    uint32_t idr_interval;

    /**
     * 帧率分子(如 30fps, 则 fps_n = 30, fps_d = 1).
     * 帧率计算方式: fps = fps_n / fps_d
     */
    uint32_t fps_n;

    /** 帧率分母. */
    uint32_t fps_d;

    /**
     * 最大 B 帧数量.
     * H.264/H.265 编码中 B 帧用于提高压缩率
     * 典型值: 0(无 B 帧), 3(平衡模式), 4-5(高压缩率)
     */
    uint32_t max_b_frames;

    /**
     * 参考帧数量.
     * H.264: 1-4
     * H.265: 1-8(更高参考帧数可提高压缩效率)
     */
    uint32_t refs;

    /**
     * 最大量化参数(QP, 值越大质量越低).
     * 取值范围 0-51(H.264), 0-63(H.265)
     */
    uint32_t qmax;

    /**
     * 最小量化参数(QP, 值越小质量越高).
     * 取值范围 0-51(H.264), 0-63(H.265)
     */
    uint32_t qmin;

    /**
     * 硬件编码预设类型.
     * 0 = 默认, 1 = 低延迟, 2 = 高质量, 3 = 高性能
     */
    uint32_t hw_preset_type;

    /**
     * 扩展参数.
     * 
     * @warning 禁止释放, 有效期受源影响.
    */
    const uint8_t *ext_data;

    /**扩展参数长度.*/
    uint32_t ext_size;

} abcdk_xpu_vcodec_params_t;


/**DNN张量.*/
typedef struct _abcdk_xpu_dnn_tensor
{
    /**索引.*/
    int index;

    /**
     * 名字.
     * 
     * @note 复制的指针, 对象销毁前有效.
    */
    const char *name_p;

    /**
     * 模式.
     * 
     * 未知：0
     * 输入：1
     * 输出：2
    */
    int mode;

    /**维度.*/
    abcdk_xpu_dims_t dims;

    /**
     * 数据.
     * 
     * @note 复制的指针, 对象销毁前有效.
     * @note 仅输出有效, 并且数据在主机内存.
    */
    const void *data_p;

} abcdk_xpu_dnn_tensor_t;

/**DNN目标.*/
typedef struct _abcdk_xpu_dnn_object
{
    /**标签.*/
    int label;

    /**评分.*/
    int score;

    /**
     * 矩形坐标.
     * 
     * 左上：pt[0]
     * 右下：pt[1]
    */
    abcdk_xpu_polygon_t rect;

    /** 旋转角度.*/
    int angle;

    /**
     * 旋转矩形坐标.
     * 
     * 左上：pt[0]
     * 右上：pt[1]
     * 右下：pt[2]
     * 左下：pt[3]
     * 
     * @note 可能超出图像范围.
     */
    abcdk_xpu_polygon_t rrect;

    /**
     * 关键点.
     * 
     * [[x,y,v],...]
    */
    int nkeypoint;
    int *kp;

    /**特征.*/
    int nfeature;
    float *ft;

    /**分割.*/
    int seg_step;
    float *seg;

    /**追踪ID.*/
    int track_id;

} abcdk_xpu_dnn_object_t;

__END_DECLS

#endif // ABCDK_XPU_TYPES_H