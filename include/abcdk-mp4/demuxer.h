/*
 * This file is part of ABCDK.
 * 
 * MIT License
 * 
 */
#ifndef ABCDK_MP4_DEMUXER_H
#define ABCDK_MP4_DEMUXER_H

#include "abcdk-util/tree.h"
#include "abcdk-mp4/atom.h"
#include "abcdk-mp4/file.h"

__BEGIN_DECLS



/**
 * 读取atom结构(遇到容器时递归读取)。
 * 
 * @param fd 文件句柄。
 * @param offset 偏移量，以0为基值。
 * @param size 最大长度，-1UL 直到文件末尾。
 * 
 * @return !NULL(0) 成功，NULL(0) 失败。
*/
abcdk_tree_t *abcdk_mp4_read_probe(int fd, uint64_t offset, uint64_t size);

/**
 * 读取FULLBOX头部。
*/
int abcdk_mp4_read_fullheader(int fd, uint8_t *ver, uint32_t *flags);

/**
 * 读取内容数据。
 * 
 * @return 0 成功，-1 失败()
*/
int abcdk_mp4_read_content(int fd, abcdk_mp4_atom_t *atom);

__END_DECLS

#endif //ABCDK_MP4_DEMUXER_H
