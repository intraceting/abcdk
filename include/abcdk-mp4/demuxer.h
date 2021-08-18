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

/**/
abcdk_tree_t *abcdk_mp4_read_probe(int fd);


__END_DECLS

#endif //ABCDK_MP4_DEMUXER_H
