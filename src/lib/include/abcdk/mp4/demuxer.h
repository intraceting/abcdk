/*
 * This file is part of ABCDK.
 * 
 * Copyright (c) 2021 The ABCDK project authors. All Rights Reserved.
 * 
 */
#ifndef ABCDK_MP4_DEMUXER_H
#define ABCDK_MP4_DEMUXER_H

#include "abcdk/util/tree.h"
#include "abcdk/mp4/atom.h"
#include "abcdk/mp4/file.h"

__BEGIN_DECLS

/**
 * 读取atom结构头部。
 *  
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_tree_t *abcdk_mp4_read_header(int fd);

/**
 * 读取FULLBOX头部。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_mp4_read_fullheader(int fd, uint8_t *ver, uint32_t *flags);

/**
 * 读取atom内容(递归读取所有子节点)。
 * 
 * @note free,skip,wide,mdat 这四种类型忽略内容。
 * 
 * @return 0 成功，-1 失败。
*/
int abcdk_mp4_read_content(int fd, abcdk_tree_t *node);

/**
 * 读取atom结构(递归读取所有子节点)。
 * 
 * @param fd 文件句柄。
 * @param offset 偏移量，以0为基值。
 * @param size 最大长度，-1UL 直到文件末尾。
 * @param stop 中断tag，NULL(0) 忽略。 
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_tree_t *abcdk_mp4_read_probe(int fd, uint64_t offset, uint64_t size, abcdk_mp4_tag_t *stop);

/**
 * 读取atom结构(递归读取所有子节点)。
 * 
 * @param stop 中断类型(大端字节序)。。 
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_tree_t *abcdk_mp4_read_probe2(int fd, uint64_t offset, uint64_t size, uint32_t stop);


__END_DECLS

#endif //ABCDK_MP4_DEMUXER_H
