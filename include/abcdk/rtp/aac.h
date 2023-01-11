/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_RTP_AAC_H
#define ABCDK_RTP_AAC_H

#include "abcdk/util/general.h"
#include "abcdk/util/queue.h"
#include "abcdk/util/bloom.h"
#include "abcdk/comm/message.h"

__BEGIN_DECLS

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

__END_DECLS

#endif //ABCDK_RTP_AAC_H