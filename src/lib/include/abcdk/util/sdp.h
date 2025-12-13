/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_UTIL_SDP_H
#define ABCDK_UTIL_SDP_H

#include "abcdk/util/general.h"
#include "abcdk/util/tree.h"
#include "abcdk/util/io.h"
#include "abcdk/util/basecode.h"
#include "abcdk/util/object.h"
#include "abcdk/util/string.h"

__BEGIN_DECLS

/** RTSP媒体基本描述信息.*/
typedef struct _abcdk_sdp_media_base
{
    /** 编码名称.*/
    abcdk_object_t *encoder;

    /** 时间速率.*/
    uint32_t clock_rate;

    /** 编码参数.*/
    abcdk_object_t *encoder_param;

    /** FMTP参数.*/
    abcdk_object_t *fmtp_param[100]; 

    /** 
     * 流地址.
     * 
     * @note 相对地址, 或绝对地址.
    */
    abcdk_object_t *control;

    /** 
     * 编码扩展.
     * 
     * @note HEVC有效.
    */
    abcdk_object_t *sprop_vps;

    /** 
     * 编码扩展.
     * 
     * @note HEVC, H264有效.
    */
    abcdk_object_t *sprop_sps;

    /** 
     * 编码扩展.
     * 
     * @note HEVC, H264有效.
    */
    abcdk_object_t *sprop_pps;

    /** 
     * 编码扩展.
     * 
     * @note HEVC, H264有效.
    */
    abcdk_object_t *sprop_sei;


} abcdk_sdp_media_base_t;

/** 分析SDP.*/
abcdk_tree_t *abcdk_sdp_parse(const char *data, size_t size);

/** 打印SDP.*/
void abcdk_sdp_dump(FILE *fp, abcdk_tree_t *sdp);

/** 
 * 在SDP中查找媒体节点.
 * 
 * @param [in] fmt 媒体格式(载荷).
 * 
 * @return !NULL(0) 成功(节点指针), NULL(0) 失败.
*/
abcdk_tree_t *abcdk_sdp_find_media(abcdk_tree_t *sdp, uint8_t fmt);

/**释放SDP媒体基本信息.*/
void abcdk_sdp_media_base_free(abcdk_sdp_media_base_t **ctx);

/**
 * 收集SDP中媒体基本信息.
 *  
 * @param [in] fmt 媒体格式(载荷).
 * 
 * @return !NULL(0) 成功, NULL(0) 失败(或未找到符合的媒体格式).
*/
abcdk_sdp_media_base_t *abcdk_sdp_media_base_collect(abcdk_tree_t *sdp,uint8_t fmt);

__END_DECLS

#endif //ABCDK_UTIL_SDP_H