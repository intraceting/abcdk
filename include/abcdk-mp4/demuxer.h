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
 * 读取MP4文件结构。
*/
abcdk_tree_t *abcdk_mp4_read_probe(int fd, int moov_only);

/**
 * 读取ftyp数据。
 * 
 * @return 0 成功，-1 失败()
*/
int abcdk_mp4_atom_read_ftyp(abcdk_mp4_atom_ftyp_t *cont, const abcdk_mp4_atom_t *atom, int fd);

/** 
 * 读取mvhd数据。
 * 
*/
int abcdk_mp4_atom_read_mvhd(abcdk_mp4_atom_mvhd_t *cont, const abcdk_mp4_atom_t *atom, int fd);

/** 
 * 读取tkhd数据。
 * 
*/
int abcdk_mp4_atom_read_tkhd(abcdk_mp4_atom_tkhd_t *cont, const abcdk_mp4_atom_t *atom, int fd);

/** 
 * 读取hdlr数据。
 * 
*/
int abcdk_mp4_atom_read_hdlr(abcdk_mp4_atom_hdlr_t *cont, const abcdk_mp4_atom_t *atom, int fd);

__END_DECLS

#endif //ABCDK_MP4_DEMUXER_H
