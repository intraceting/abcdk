/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_RTP_AAC_H
#define ABCDK_RTP_AAC_H

#include "abcdk/util/general.h"
#include "abcdk/comm/queue.h"
#include "abcdk/util/bloom.h"

__BEGIN_DECLS

/**
 * AAC数据包还原。
 * 
 * @param [in] size_length 数据包长度的长度(bits)。可变参数为其它动态字段长度的长度，< 0 结束。 
 *
 * @return 1 已还原，0 需要更多数据，-1 有错误发生，-2 未支持的类型。
 */
int abcdk_rtp_aac_revert(const void *data, size_t size, abcdk_comm_queue_t *q, int size_length, ...);

__END_DECLS

#endif //ABCDK_RTP_AAC_H