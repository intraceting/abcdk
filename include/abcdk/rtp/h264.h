/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_RTP_H264_H
#define ABCDK_RTP_H264_H

#include "abcdk/util/general.h"
#include "abcdk/util/queue.h"
#include "abcdk/util/bloom.h"
#include "abcdk/util/receiver.h"

__BEGIN_DECLS

/**
 * H264数据包还原。
 *
 * @return 1 已还原，0 需要更多数据，-1 有错误发生，-2 未支持的类型。
 */
int abcdk_rtp_h264_revert(const void *data, size_t size, abcdk_queue_t *q);

__END_DECLS

#endif //ABCDK_RTP_H264_H