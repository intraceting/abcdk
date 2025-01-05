/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) intraceting<intraceting@outlook.com>
 * 
 */
#ifndef ABCDK_RTP_RTP_H
#define ABCDK_RTP_RTP_H

#include "abcdk/util/general.h"
#include "abcdk/util/queue.h"
#include "abcdk/util/bloom.h"
#include "abcdk/util/receiver.h"

__BEGIN_DECLS

/**RTP头部。*/
typedef struct _abcdk_rtp_header
{
    /**
     * RTP版本号，2bits。
     * 
     * 当前版本号为2。
    */
    uint8_t version;

    /**
     * 填充标志，1bit。
     * 
     * 如果P=1，则在该报文的尾部填充一个或多个额外的八位组，它们不是有效载荷的一部分。
    */
    uint8_t padding;

    /**
     * 扩展标志，1bit。
     * 
     * 如果X=1，则在RTP报头后跟有一个扩展报头。
    */
    uint8_t  extension;

    /** 
     * CSRC计数器，4bits。
     * 
     * 指示CSRC标识符的个数，每个占32bits。
    */
    uint8_t  csrc_len;

    /** 
     * 标记，1bit。
     * 
     * 按载荷自定义，目的在于允许重要事件在包流中标记出来。
     * 对于视频，标记一帧的结束；对于音频，标记会话的开始。
    */
    uint8_t  marker;

    /**
     * 载荷类型，7bits。
    */
    uint8_t  payload;

    /**
     * 序列号，16bits。
     * 
     * 序列号的初始值随机生成。
     * 每发送一个报文，序列号增加1。
     * 用于检查是否丢包和排序。
    */
    uint32_t seq_no;

    /**
     * 时间戳，32bits。
    */
    uint32_t timestamp;

    /** 
     * 同步信源(SSRC)标识符，32bits。
     * 
     * 该标识符随机生成。
     * 同路多个信源之间标识符不能相同。
    */
    uint32_t ssrc;

    /**
     * 提供信源(CSRC)标识符，0~512bits。
     * 
     * 每个CSRC标识符占32bits。
     * 每个CSRC标识了包含在RTP报文有效载荷中的所有提供信源。
    */
    uint32_t csrc[16];

} abcdk_rtp_header_t;

/**序列化。*/
void abcdk_rtp_header_serialize(const abcdk_rtp_header_t *hdr, void *data, size_t size);

/**反序列化。*/
void abcdk_rtp_header_deserialize(const void *data, size_t size, abcdk_rtp_header_t *hdr);

/**
 * AAC数据包还原。
 * 
 * @note RTP的AAC封包有8个可变长度的字段，这里仅支持两个字段。
 * 
 * @param [in] size_bits 数据包长度的长度(bits)。
 * @param [in] index_bits 数据包索引的长度(bits)。
 *
 * @return 1 已还原，0 需要更多数据，-1 有错误发生，-2 未支持的类型。
 */
int abcdk_rtp_aac_revert(const void *data, size_t size, abcdk_queue_t *q, int size_bits, int index_bits);

/**
 * H264数据包还原。
 *
 * @return 1 已还原，0 需要更多数据，-1 有错误发生，-2 未支持的类型。
 */
int abcdk_rtp_h264_revert(const void *data, size_t size, abcdk_queue_t *q);

/**
 * H265数据包还原。
 *
 * @return 1 已还原，0 需要更多数据，-1 有错误发生，-2 未支持的类型。
 */
int abcdk_rtp_hevc_revert(const void *data, size_t size, abcdk_queue_t *q);


__END_DECLS

#endif //ABCDK_RTP_RTP_H