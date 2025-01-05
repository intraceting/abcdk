/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021-Present intraceting<intraceting@outlook.com>
 *
 */
#ifndef ABCDK_VIDEO_H264_H
#define ABCDK_VIDEO_H264_H

#include "abcdk/util/general.h"
#include "abcdk/util/bit.h"
#include "abcdk/util/object.h"
#include "abcdk/video/h2645.h"

__BEGIN_DECLS

/** H264扩展数据结构。*/
typedef struct _abcdk_h264_extradata
{
    /**
     * 版本号，8bits。
     * 
     * 当前版本号为1。
    */
    uint8_t version;

    /*
     * 概述，8bits。
    */
    uint8_t avc_profile;

    /*
     * 兼容性，8bits。
    */
    uint8_t avc_compatibility;

    /*
     * 分级，8bits。
    */
    uint8_t avc_level;

    /*
     * 预留1，6bits。
     * 
     * 全是1。
    */
    uint8_t reserve1;

    /**
     * NAL长度的大小，2bits。
     * 
     * 在实际的大小减去1。
     * 
     * 如：3=4-1；2=3-1；1=2-1。
    */
    uint8_t nal_length_size;

    /**
     * 预留2，3bits。
     * 
     * 全是1。
    */
    uint8_t reserve2;

    /**
     * SPS数量,5bits。
    */
    uint8_t sps_num;

    /** 
     * SPS。
     * 
     * 串行格式：len(16bits)+data(nBytes)
     * 
     * @note 谁使用，谁释放。
    */
    abcdk_object_t *sps;

    /**
     * PPS数量,8bits。
    */
    uint8_t pps_num;

    /** 
     * PPS。
     * 
     * 串行格式：len(16bits)+data(nBytes)
     * 
     * @note 谁使用，谁释放。
    */
    abcdk_object_t *pps;

}abcdk_h264_extradata_t;

/**
 * 清理。
 * 
 * @warning 未初始环境调用后，产生的结果不可预知。
*/
void abcdk_h264_extradata_clean(abcdk_h264_extradata_t *extradata);

/**序列化。*/
ssize_t abcdk_h264_extradata_serialize(const abcdk_h264_extradata_t *extdata, void *data, size_t size);

/**反序列化。*/
void abcdk_h264_extradata_deserialize(const void *data, size_t size, abcdk_h264_extradata_t *extdata);

/**
 * 判断是否包括关键帧。
 * 
 * @return 关键帧数量。
*/
int abcdk_h264_idr(const void *data, size_t size);

__END_DECLS

#endif // ABCDK_VIDEO_H264_H