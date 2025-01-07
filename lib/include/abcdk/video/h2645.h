/*
 * This file is part of ABCDK.
 *
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 *
 */
#ifndef ABCDK_VIDEO_H2645_H
#define ABCDK_VIDEO_H2645_H

#include "abcdk/util/defs.h"
#include "abcdk/util/bit.h"

__BEGIN_DECLS

/**
 * 查找起始码。
 * 
 * @param [in] b 起始指针。
 * @param [in] e 结束指针。
 * @param [in out] msize 起始码长度。
 * 
 * @return >=0 成功(当前位置到起始码开始的偏移量)，< 0 不存在。
*/
ssize_t abcdk_h2645_find_start_code(const void *b, const void *e,int *msize);

/**
 * 分包，拆包。 
 * 
 * @param [in out] next 输入起始指针，输出下一个包的指针(可能超过结束指针)。
 * @param [in] e 结束指针。
 * 
 * @return !NULL(0) 数据包的指针(跳过起始码的)，NULL(0) 末尾。
*/
const void *abcdk_h2645_packet_split(void **next,const void *e);

/** 
 * avcc或hvcc转字节流。
 * 
 * 本质上就是把帧的长度信息，替换为起始码。
 * 
 * @note 不会在I帧前添加扩展信息。
 * 
 * @param [in] len_size 起始码长度。
*/
void abcdk_h2645_mp4toannexb(void *data,size_t size,int len_size);

__END_DECLS

#endif //ABCDK_VIDEO_H2645_H
