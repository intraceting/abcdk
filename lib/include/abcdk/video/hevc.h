/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#ifndef ABCDK_VIDEO_HEVC_H
#define ABCDK_VIDEO_HEVC_H

#include "abcdk/util/general.h"
#include "abcdk/util/bit.h"
#include "abcdk/util/object.h"
#include "abcdk/video/h2645.h"

__BEGIN_DECLS

/** HEVC扩展数据结构。*/
typedef struct _abcdk_hevc_extradata
{
    /**
     * 版本号，8bits。
     *
     * 当前版本号为1。
     */
    uint8_t version;

    /**
     * 概述，2bits。
     *
     */
    uint8_t general_profile_space;

    /**
     * 1bits。
     *
     */
    uint8_t general_tier_flag;

    /**
     * 5bits。
     *
     */
    uint8_t general_profile_idc;

    /**
     * 32bits。
     *
     */
    uint32_t general_profile_compatibility_flags;

    /**
     * 48bits。
     *
     */
    uint64_t general_constraint_indicator_flags;

    /**
     * 8bits。
     *
     */
    uint8_t general_level_idc;

    /*
     * 预留1，4bits。
     *
     * 全是1。
     */
    uint8_t reserve1;

    /**
     * 12bits。
     *
     */
    uint16_t min_spatial_segmentation_idc;

    /*
     * 预留2，6bits。
     *
     * 全是1。
     */
    uint8_t reserve2;

    /**
     * 2bits。
     *
     */
    uint8_t parallelism_type;

    /*
     * 预留2，6bits。
     *
     * 全是1。
     */
    uint8_t reserve3;

    /**
     * 2bits。
     *
     */
    uint8_t chroma_format;

    /*
     * 预留4，5bits。
     *
     * 全是1。
     */
    uint8_t reserve4;

    /**
     * 3bits。
     *
     */
    uint8_t bit_depth_luma_minus8;

    /*
     * 预留5，5bits。
     *
     * 全是1。
     */
    uint8_t reserve5;

    /**
     * 3bits。
     *
     */
    uint8_t bit_depth_chroma_minus8;

    /**
     * 16bits。
     *
     */
    uint16_t avg_frame_rate;

    /**
     * 2bits。
     *
     */
    uint8_t constant_frame_rate;

    /**
     * 3bits。
     *
     */
    uint8_t num_temporal_layers;

    /**
     * 1bit。
     *
     */
    uint8_t temporal_id_nested;

    /**
     * NAL长度的大小，2bits。
     *
     * 在实际的大小减去1。
     *
     * 如：3=4-1；2=3-1；1=2-1。
     */
    uint8_t nal_length_size;

    /**
     * NAL数量,8bits。
     */
    uint8_t nal_array_num;

    /**
     * NAL数组。
     */
    struct _nal_array
    {
        /*
         * 1bit。
         */
        uint8_t array_completeness;

        /**
         * 预留1，1bit。
         *
         * 全是0。
         */
        uint8_t reserve1;

        /**
         * NAL类型，6bit。
         */
        uint8_t unit_type;

        /**
         * NAL数量,16bits。
         */
        uint16_t nal_num;

        /**
         * NAL。
         *
         * 串行格式：len(16bits)+data(nBytes)
         *
         * @note 谁使用，谁释放。
         */
        abcdk_object_t *nal;

    } nal_array[256];

} abcdk_hevc_extradata_t;

/**
 * 清理。
 * 
 * @warning 未初始环境调用后，产生的结果不可预知。
*/
void abcdk_hevc_extradata_clean(abcdk_hevc_extradata_t *extdata);

/**序列化。*/
ssize_t abcdk_hevc_extradata_serialize(const abcdk_hevc_extradata_t *extradata, void *data, size_t size);

/**反序列化。*/
void abcdk_hevc_extradata_deserialize(const void *data, size_t size, abcdk_hevc_extradata_t *extradata);

/**
 * 判断是否包括关键帧。
 * 
 * @return 关键帧数量。
*/
int abcdk_hevc_irap(const void *data, size_t size);

__END_DECLS

#endif // ABCDK_VIDEO_HEVC_H